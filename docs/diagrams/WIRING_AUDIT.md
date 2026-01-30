# T4DM Neural Wiring Audit

**Reviewer**: Geoffrey Hinton (AI Architect Agent)
**Date**: 2026-01-29
**Scope**: All neuroscience-related mermaid diagrams in `/mnt/projects/t4d/t4dm/docs/diagrams/`

---

## Executive Summary

The diagrams demonstrate a genuinely thoughtful attempt to map biological neural circuits onto a software memory system. The hippocampal trisynaptic pathway, neuromodulator orchestra, and sleep consolidation chains are structurally sound in their broad strokes. However, there are systematic issues: missing inhibitory interneurons throughout, absent feedback projections that are critical for biological function, some incorrect temporal orderings in sleep oscillation coupling, and an oversimplified treatment of neuromodulator interactions. The Forward-Forward and capsule routing diagrams are reasonable adaptations but conflate software design choices with neural mechanisms in ways that could mislead.

Below I evaluate each diagram individually.

---

## 1. hippocampal_circuit.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/hippocampal_circuit.mermaid`

### Correct Wiring
- EC to DG via perforant path: correct
- DG to CA3 via mossy fibers: correct
- CA3 recurrent collaterals: correct, this is the autoassociative attractor network
- CA3 to CA1 via Schaffer collaterals: correct
- EC direct path to CA1: correct (temporoammonic pathway)
- CA1 novelty detection via EC vs CA3 comparison: correct, matches Hasselmo & Stern (2006)
- ACh modulating encoding (high) vs retrieval (low): correct, matches Hasselmo (1999)

### Missing Connections
- **CA1 output to subiculum**: The diagram goes CA1 to OUT ("Output to Cortex") but skips the subiculum entirely. CA1 projects primarily to subiculum, which then projects to EC deep layers. This is a critical relay (Amaral & Witter, 1989).
- **EC Layer II vs Layer III distinction**: EC Layer II projects to DG/CA3 (perforant path), while EC Layer III projects to CA1 (temporoammonic). The diagram collapses these into a single EC node.
- **Mossy cell feedback loop**: DG has mossy cells that project back to distant granule cells, providing a feedback/lateral inhibition circuit. Missing entirely.
- **Basket cell inhibition in DG**: Pattern separation depends critically on GABAergic basket cells providing lateral inhibition. The diagram mentions "winner-take-all" in DG_OP but has no inhibitory interneuron nodes.
- **CA1 to EC deep layers feedback**: The return path completing the hippocampal loop (CA1 to EC Layer V/VI) is absent. Only "OUT" is shown.
- **DA projection to CA1**: DA from VTA/LC gates CA1 LTP, not just CA3. The diagram only shows DA modulating CA3_OP.

### Wrong Connections
- **MODE feedback to DG and CA3**: The diagram shows novelty score feeding back to DG (encoding) and CA3 (retrieval) as if the mode decision directly controls these regions. In biology, this is mediated by ACh from the medial septum/diagonal band, not by a direct "mode" signal from CA1. The diagram partially captures this with the ACh node but then also has the direct MODE arrows, creating a confusing dual pathway.
- **Grid cells to Place cells to DG**: Grid cells (medial EC) project to DG through the perforant path, not via place cells as intermediaries. Place cells are in CA1/CA3, downstream of grid cell input. The arrow GRID to PLACE to DG reverses the actual hierarchy. Grid cells are in EC; place cells emerge in hippocampus proper.

### Recommendations
1. Add subiculum as an explicit node between CA1 and output
2. Split EC into Layer II and Layer III nodes
3. Remove the GRID to PLACE to DG chain; replace with medial EC (grid cells) to DG and show place cells as emergent in CA1/CA3
4. Add at least one GABAergic interneuron node in DG for pattern separation
5. Remove the direct MODE to DG/CA3 arrows; let ACh handle this exclusively

---

## 2. hpc_trisynaptic.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/hpc_trisynaptic.mermaid`

### Correct Wiring
- EC Layer II to granule cells (perforant path): correct
- EC Layer III to CA1 (temporoammonic): correct, well done distinguishing from the first diagram
- Granule cells to CA3 via mossy fibers with "giant synapses" annotation: correct and good detail (Henze et al., 2000)
- CA3 recurrent collaterals: correct
- CA3 to CA1 via Schaffer collaterals: correct
- CA1 to subiculum to EC Layer V to neocortex: correct and complete output chain
- Mossy cells and hilar interneurons in DG: good inclusion
- Inhibitory interneurons in CA1: correctly placed
- NE gain control on CA3: correct (Harley, 2007)

### Missing Connections
- **EC Layer II also projects to CA3**: Not just via DG. There is a direct EC II to CA3 projection that bypasses DG (the "direct perforant path to CA3"). This matters for models because it means CA3 receives both processed (via DG) and raw (direct from EC) input.
- **CA3 to lateral septum projection**: CA3 projects to the lateral septum, which connects to the hypothalamus. Not critical for memory per se but relevant for the neuromodulatory feedback loop.
- **Hilar interneuron connections**: The diagram includes hilar interneurons (HILUS) but does not show any connections from them. They should inhibit granule cells (feedback inhibition) and receive input from mossy fibers.
- **Mossy cell connections**: Mossy cells (MOSSY) are shown but unconnected. They receive input from granule cells and CA3 and project back to granule cells in distant lamellae.
- **CA1 inhibitory interneurons source**: INH_CA1 inhibits PYR_CA1, which is correct, but where does it receive input? It should receive input from both Schaffer collaterals and the temporoammonic pathway (feedforward inhibition).

### Wrong Connections
- **DA gates LTP at CA1**: This is partially correct but incomplete. DA (from VTA) does gate LTP at CA1 (Lisman & Grace, 2005), but it also strongly gates CA3 plasticity and DG plasticity. Showing it only at CA1 is misleading.

### Recommendations
1. Add direct EC II to CA3 connection (bypassing DG)
2. Connect hilar interneurons: mossy fibers to HILUS, HILUS inhibits GC
3. Connect mossy cells: GC to MOSSY, MOSSY to distant GC
4. Show input sources for CA1 inhibitory interneurons
5. Add DA modulation to DG and CA3 as well

---

## 3. vta_circuit.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/vta_circuit.mermaid`

### Correct Wiring
- VTA inputs from lateral hypothalamus, LDT/PPTg, PFC, RMTg: correct and well-chosen (Morales & Margolis, 2017)
- RMTg as GABAergic brake on DA neurons: correct (Jhou et al., 2009)
- DA neuron percentages (~60% DA, ~35% GABA, ~5% Glu): approximately correct
- Tonic (3-5 Hz) vs phasic (15-25 Hz) firing modes: correct (Grace & Bunney, 1984)
- RPE computation (R - V = delta): correct, Schultz (1997) formulation
- Projections to NAc, dlPFC, hippocampus, amygdala: correct target set
- GABA interneuron local inhibition of DA neurons: correct
- Positive RPE triggers burst, negative RPE triggers pause: correct

### Missing Connections
- **Glutamate neurons (GLU_NEURON) have no connections**: The diagram includes them but they are orphaned. VTA glutamate neurons project locally to DA and GABA neurons and also to NAc and PFC (Yamaguchi et al., 2011). They play a role in aversive signaling.
- **NAc feedback to VTA**: The NAc sends GABAergic projections back to VTA (direct and indirect pathways). This is a critical feedback loop for reward learning. Missing entirely.
- **Amygdala input to VTA**: The amygdala (especially CeA and BLA) sends projections to VTA that convey salience/threat information. Only shown as output, not input.
- **Habenula input**: The lateral habenula is a major source of inhibitory control over VTA DA neurons (via RMTg). It encodes negative RPE and drives DA pauses. Not shown.
- **D2 autoreceptor feedback**: DA neurons have D2 autoreceptors that provide negative feedback on their own firing. This self-inhibition is critical for tonic firing regulation.
- **Substantia Nigra pars compacta (SNc)**: VTA is shown in isolation, but SNc is the other major midbrain DA source projecting to dorsal striatum. The diagram could at least note this.

### Wrong Connections
- **RPE to TONIC labeled as "-delta to pause"**: This is slightly misleading. Negative RPE causes a *pause* in tonic firing, not a switch *to* tonic mode. Tonic firing is the baseline state. The arrow should be "negative delta causes pause in tonic firing" rather than implying tonic is a response mode.

### Recommendations
1. Connect glutamate neurons: GLU to DA_NEURON (local excitation), GLU to NAc
2. Add NAc to VTA feedback (GABAergic)
3. Add lateral habenula as input (via RMTg or direct)
4. Add D2 autoreceptor self-inhibition arrow on DA_NEURON
5. Clarify the tonic/phasic relationship: tonic is baseline, phasic burst is superimposed on tonic

---

## 4. neuromod_orchestra.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/neuromod_orchestra.mermaid`

### Correct Wiring
- Four neuromodulator sources (VTA/DA, Raphe/5-HT, LC/NE, BF/ACh): correct canonical set
- RPE drives DA, novelty drives NE, attention drives ACh: correct mappings
- DA effects (learning rate, vigor, goal pursuit): correct
- 5-HT effects (discount rate, patience, impulse control): correct, matches Doya (2002) framework
- NE effects (gain modulation, arousal, focus width): correct, matches Aston-Jones & Cohen (2005)
- ACh effects (encoding mode, retrieval mode, attention): correct, matches Hasselmo (2006)
- Cross-modulation DA inhibits 5-HT: partially correct

### Missing Connections
- **5-HT inhibits DA (via VTA)**: The diagram shows DA inhibits 5-HT, but the reciprocal is more established. Raphe 5-HT neurons project to VTA and inhibit DA neurons. This is the opponent process (Daw et al., 2002). The direction shown (DA inhibits 5-HT) exists but is less prominent.
- **NE-ACh interaction**: LC-NE and BF-ACh have important interactions. NE promotes ACh release in cortex (via alpha-1 receptors on BF neurons). Missing.
- **GABA entirely absent**: GABA is not a neuromodulator in the traditional sense, but inhibitory interneurons are critical for implementing the effects of all four modulators. The orchestra should at least reference GABAergic tone.
- **Histamine (TMN)**: The tuberomammillary nucleus produces histamine, which is important for the wake/sleep and arousal states. Not mentioned.
- **Orexin/hypocretin**: From the lateral hypothalamus, orexin stabilizes waking states and interacts with all four modulator systems. Important for the CONSOLIDATE state.
- **Waiting/patience as 5-HT input**: This is correct per Miyazaki et al. (2014) but oversimplified. 5-HT is driven by many signals beyond "waiting" including uncertainty, aversive prediction, and behavioral inhibition.

### Wrong Connections
- **NE modulates DA**: The diagram shows NE_LEVEL modulating DA_LEVEL. While LC does project to VTA, the primary interaction is that NE modulates DA *target regions* (e.g., NE in PFC modulates DA effects there), not DA production directly. The arrow implies NE controls DA level, which overstates the interaction.

### Recommendations
1. Reverse or make bidirectional the DA-5HT interaction; emphasize 5-HT inhibition of DA
2. Add NE to ACh interaction (NE promotes ACh release)
3. Add GABA as a fifth element (or note its role)
4. Rename "Waiting/Patience" to something broader like "Behavioral state / Uncertainty"
5. Distinguish between NE modulating DA targets vs NE modulating DA production

---

## 5. neuromodulator_pathways.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/neuromodulator_pathways.mermaid`

### Correct Wiring
- Substantia Nigra included alongside VTA: good addition over the orchestra diagram
- VTA tonic to striatum, burst to PFC: approximately correct (though both tonic and phasic reach both targets)
- DRN 5-HT projections to PFC, HPC, amygdala: correct broad projection pattern
- LC NE tonic vs phasic modes with correct frequencies: correct (Aston-Jones & Cohen, 2005)
- NBM (Nucleus Basalis of Meynert) ACh to HPC and PFC: correct
- VTA-DRN opponent interaction: correct
- LC-VTA arousal sync: correct

### Missing Connections
- **VTA to hippocampus**: VTA DA to HPC is shown in the VTA circuit diagram but missing here. This is a critical pathway for memory encoding gating (Lisman & Grace, 2005).
- **VTA to amygdala**: Also missing here but present in the VTA diagram. Inconsistent.
- **LC to amygdala**: Shown, which is correct.
- **NBM to amygdala**: BF/NBM also projects cholinergic fibers to the amygdala. Missing.
- **DRN to striatum**: 5-HT innervation of the striatum is dense and functionally important (impulse control, patience in reward waiting). Missing.
- **LC projections to almost everything**: LC-NE is the most broadly projecting of all neuromodulator systems. It should project to striatum and amygdala as well. Only PFC, HPC, AMY shown (AMY is shown, good).
- **Feedback from PFC to all sources**: PFC sends top-down regulatory projections to VTA, LC, DRN, and NBM. This is critical for cognitive control of neuromodulation. Entirely missing.
- **Salience as DRN input**: Salience is not a primary driver of 5-HT. Uncertainty or aversive prediction would be more accurate. The amygdala and PAG drive DRN more than abstract "salience."

### Wrong Connections
- **VTA tonic specifically to striatum and burst specifically to PFC**: This is an oversimplification. Both tonic and phasic DA reach both targets. The distinction is in how the targets respond (D1 vs D2 receptor populations in striatum respond differentially to tonic vs phasic). The diagram implies anatomical segregation of firing modes to targets, which is incorrect.

### Recommendations
1. Add VTA to HPC and VTA to AMY projections
2. Add DRN to striatum
3. Add PFC feedback to all four neuromodulator sources
4. Fix the tonic/phasic to target mapping: both modes reach both targets
5. Change DRN input from "Salience" to "Aversive prediction / Uncertainty"

---

## 6. adenosine_homeostasis.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/adenosine_homeostasis.mermaid`

### Correct Wiring
- ATP breakdown to adenosine accumulation during wake: correct
- A1 receptor (inhibitory) and A2A receptor (sleep-promoting): correct
- A1 inhibits wake-promoting neurons (BF, TMN): correct
- A2A activates VLPO (sleep-promoting): correct (Scammell et al., 2001)
- VLPO GABAergic inhibition of arousal systems: correct (Saper et al., 2005)
- Two-process model (Process S homeostatic, Process C circadian): correct (Borbely, 1982)
- Caffeine antagonism at both A1 and A2A: correct
- Glymphatic clearance during sleep: correct (Xie et al., 2013)

### Missing Connections
- **Flip-flop switch**: The VLPO and arousal systems (TMN, LC, DRN, VTA) mutually inhibit each other, creating a bistable flip-flop switch (Saper et al., 2001). The diagram shows VLPO inhibiting arousal but not the reciprocal: arousal systems (especially NE from LC, 5-HT from DRN) inhibit VLPO during waking. This mutual inhibition is what creates sharp state transitions rather than gradual ones.
- **Orexin stabilization**: Orexin from the lateral hypothalamus stabilizes the flip-flop in the wake position. Without it you get narcolepsy. Not shown.
- **SCN (suprachiasmatic nucleus) driving Process C**: Process C is mentioned but its source (the SCN master circadian clock) is not shown.
- **Astrocytic adenosine release**: Much of the adenosine that drives sleep pressure is released by astrocytes, not just from neuronal ATP breakdown (Halassa et al., 2009). The diagram only shows the neuronal ATP pathway.

### Wrong Connections
- No outright wrong connections. This diagram is one of the more accurate ones.

### Recommendations
1. Add mutual inhibition: arousal systems inhibit VLPO (flip-flop switch)
2. Add orexin/lateral hypothalamus stabilizing the wake state
3. Add SCN as the source of Process C
4. Note astrocytic contribution to adenosine

---

## 7. sleep_cycle.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/sleep_cycle.mermaid`

### Correct Wiring
- Wake to NREM to REM cycling: correct
- N1 to N2 to N3 progression: correct
- ~90 minute cycle: correct
- Sleep spindles in N2 (12-15 Hz): correct
- Slow waves in N3 (labeled as SWR 0.5-2 Hz): partially correct (see below)
- REM theta dominance, ACh peak, muscle atonia: correct
- Thalamo-cortical loops in N2: correct

### Missing Connections
- **NREM to Wake transition during N1/N2**: The diagram only shows NREM to Wake via "Alarm/threshold" but light NREM (N1/N2) transitions to wake are common and not just alarm-driven.
- **REM to NREM more common than REM to Wake**: Most sleep cycles go REM back to NREM (N2), not REM to Wake. The natural awakening from REM typically happens only in the last cycle of the night. The diagram implies equal probability.

### Wrong Connections
- **N3 note says "SWR Events (0.5-2 Hz)"**: Sharp-wave ripples are 150-250 Hz hippocampal events, not 0.5-2 Hz. The 0.5-2 Hz events are *slow oscillations* (or slow-wave activity/delta waves). SWRs occur during NREM but they are nested *within* the slow oscillation up-states. The note conflates slow oscillations with SWRs. This is a significant error.
- **"Glymphatic peak flow" in N3**: While glymphatic clearance does peak during NREM sleep, placing it specifically in N3 is approximately correct but the evidence suggests it occurs throughout NREM, not exclusively in N3 (Hablitz et al., 2020).

### Recommendations
1. Fix N3 note: change "SWR Events (0.5-2 Hz)" to "Slow Oscillations (0.5-1 Hz)" and add "SWRs (150-250 Hz) nested in up-states"
2. Add N3 to N2 lightening transition (already shown, good)
3. Indicate that early night is NREM-dominant and late night is REM-dominant

---

## 8. sleep_subsystems.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/sleep_subsystems.mermaid`

### Correct Wiring
- Adenosine accumulation driving sleep threshold: correct
- SWR detection at 80-120 Hz: slightly low but acceptable (typically 150-250 Hz in rodents, 80-120 Hz in some human studies)
- Glymphatic system sleep-gated (NREM only): correct
- Thalamic generation of spindles: correct
- Spindle-delta coupling: correct

### Missing Connections
- **Spindle-SWR temporal coupling**: The diagram shows SP_COUP (spindle-delta coupling) connecting to SWR_REP (replay), which is directionally correct but the mechanism is wrong. Spindles do not directly trigger replay. The slow oscillation up-state triggers both spindles (via thalamus) and SWRs (via hippocampus), and the temporal coupling between spindles and SWRs is coordinated by the slow oscillation, not by spindles driving SWRs.
- **Cortical slow oscillation entirely missing**: This is the conductor of the SO-spindle-SWR hierarchy. Its absence means the hierarchical nesting cannot be properly represented.
- **Adenosine threshold crossing does not directly trigger SWR**: The arrow from AD_THR to SWR subgraph implies adenosine triggers SWRs. Adenosine triggers sleep onset; SWRs then occur during NREM as a consequence of the sleep state, not directly from adenosine.

### Wrong Connections
- **SP_MEM to SWR_CON**: Spindle-mediated synaptic consolidation feeding into SWR consolidation reverses the actual direction. SWRs replay hippocampal content, and spindles provide the cortical plasticity window for that content to be consolidated. The flow should be SWR_REP provides content, spindles provide the plasticity window, and together they drive consolidation.

### Recommendations
1. Add cortical slow oscillation as the master conductor
2. Restructure: SO triggers both spindles and SWRs; SWRs are nested in spindle troughs
3. Remove direct AD_THR to SWR link; add sleep state as intermediary
4. Reverse SP_MEM to SWR_CON relationship

---

## 9. spindle_ripple_coupling.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/spindle_ripple_coupling.mermaid`

### Correct Wiring
- Slow oscillation (0.5-1 Hz) with up/down states: correct
- Cortical origin of slow oscillation: correct
- TRN (thalamic reticular nucleus) generating spindles: correct (Steriade et al., 1993)
- TRN to thalamocortical neurons: correct
- CA3 sharp wave generating CA1 ripple: correct
- Memory replay at 20x compression: correct (approximately, varies 10-20x, Lee & Wilson, 2002)
- Hierarchical nesting concept (SO nests spindles, spindles nest ripples): correct
- LTP and LTD windows during spindle phases: correct (Ngo et al., 2013)

### Missing Connections
- **Cortical feedback to thalamus**: The slow oscillation originates in cortex and the up-state drives TRN, which is shown. But the cortical down-to-up transition is also shaped by thalamocortical input. The diagram shows a unidirectional cortex to thalamus to cortex chain, but in reality there is strong bidirectional thalamocortical coupling.
- **Hippocampal sharp waves are not triggered by spindles**: The diagram shows SIGMA (spindle power) to CA3, implying spindles trigger SWRs. The actual mechanism is more nuanced: SO up-states trigger both spindles (via thalamus) and SWRs (via hippocampal reactivation). The temporal nesting means SWRs tend to occur during spindle troughs, but spindles do not cause SWRs. This is a correlation-causation error.
- **Neocortical target of replay**: Where does the replayed content go? The diagram shows replay but not its cortical target for consolidation.

### Wrong Connections
- **SIGMA to CA3**: As noted above, spindle sigma power does not drive CA3 sharp waves. The SO up-state independently drives both. Remove this arrow.
- **Nesting description says "Spindle Troughs nest ripples"**: The evidence is mixed. Staresina et al. (2015) showed SWRs are nested in the troughs of spindles in humans, while other studies show they can occur at spindle peaks. "Spindle troughs nest ripples" is one finding but stating it as the canonical pattern is debatable. The key point is that ripples are temporally coupled to spindles, coordinated by the SO.

### Timing Issues
- **T3 says "t+100ms: SWR nested"**: This is relative to what? If relative to T2 (spindle peak at t+200ms), then SWR at t+300ms total makes sense. But the label "t+100ms" is ambiguous. Should be "t+300ms" if absolute from SO up-state, or labeled relative to spindle onset.

### Recommendations
1. Remove SIGMA to CA3 arrow; add SO up-state triggering hippocampal reactivation directly
2. Add bidirectional thalamocortical coupling
3. Add neocortical consolidation target for replay content
4. Fix timing labels to be consistently absolute or relative
5. Soften "Spindle Troughs nest ripples" to "SWRs temporally coupled to spindle oscillation"

---

## 10. swr_replay.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/swr_replay.mermaid`

### Correct Wiring
- SO triggers CA3 burst: correct
- CA3 burst generates ripple: correct
- Ripple generates CA1 firing (compressed replay): correct
- 20x time compression: correct
- Recent episodes replayed with high priority: correct (Kudrimoti et al., 1999)
- Remote memories interleaved: correct, this is critical for avoiding catastrophic interference (McClelland et al., 1995)
- Thalamocortical coordination with up/down states: correct
- SWR-triggered LTP (HPC to NCX): correct
- Spindle-coupled plasticity: correct

### Missing Connections
- **Reverse replay**: SWR replay can be forward or reverse (Foster & Wilson, 2006). Reverse replay is particularly associated with reward learning. Not mentioned.
- **Reward modulation of replay**: Which episodes get replayed is strongly modulated by prior reward/DA signaling (Lansink et al., 2009). The diagram shows priority based on recency but not reward.
- **CA1 to entorhinal cortex pathway**: The replay content must travel from CA1 to EC to reach neocortex. This intermediate step is missing.
- **Inhibitory interneurons gating SWR initiation**: PV+ interneurons in CA3/CA1 are critical for shaping the ripple oscillation. Their transient silence allows the sharp wave, followed by rapid synchronized firing creating the ripple (Buzsaki, 2015).

### Wrong Connections
- **SO triggers CA3_BURST directly**: The mechanism is more indirect. During NREM, the cortical SO up-state provides excitatory drive to hippocampus via EC, which facilitates CA3 population bursts. The diagram implies a direct SO-to-CA3 connection, but the path goes through EC.

### Recommendations
1. Add EC as intermediary between SO/cortex and CA3
2. Add reverse replay as an option
3. Add reward/DA modulation of replay priority
4. Add PV+ interneuron role in ripple generation
5. Show CA1 to EC to neocortex output path

---

## 11. capsule_routing.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/capsule_routing.mermaid`

### Correct Wiring (relative to Hinton's capsule network papers)
- Primary capsules (8-dim pose vectors): correct, matches Sabour, Fross & Hinton (2017)
- Transformation matrices W_ij for predictions: correct
- Routing-by-agreement iterative process: correct
- Coupling coefficients via softmax: correct
- Squash nonlinearity: correct
- Agreement as dot product of prediction and output: correct
- Length = probability, direction = pose: correct interpretation

### Missing Connections
- **Reconstruction loss**: The original CapsNet paper uses a reconstruction decoder as a regularizer. This is absent from the diagram but is important for the capsule representations to be meaningful.
- **Number of routing iterations**: The diagram shows the iterative loop but does not specify that typically 3 iterations are used. Minor but worth noting.

### Wrong Connections
- **Neuromodulator modulation (NTMod subgraph)**: This is a creative addition but has no basis in the capsule network literature. DA modulating routing temperature, NE modulating squash threshold, ACh modulating mode, and 5-HT modulating routing patience are all novel design choices, not established capsule network features. They are not "wrong" per se since this is a software system, but labeling them alongside the standard capsule architecture conflates established theory with speculative extensions. They should be clearly marked as T4DM extensions.

### Recommendations
1. Add reconstruction decoder branch
2. Clearly separate "Standard CapsNet" from "T4DM Neuromodulator Extensions"
3. Note routing iteration count (typically 3)

---

## 12. ff_nca_coupling.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/ff_nca_coupling.mermaid`

### Correct Wiring (relative to Hinton 2022 Forward-Forward paper)
- Positive phase (real data) maximizes goodness: correct
- Negative phase (synthetic/corrupted data) minimizes goodness: correct
- Goodness function G = sum(h^2): correct, this is the sum of squared activations per layer
- Threshold theta separating positive from negative: correct

### Missing Connections
- **Layer-local learning**: The key insight of Forward-Forward is that each layer learns independently using only local information. The diagram does not emphasize this locality. There should be multiple layers each with their own goodness computation.
- **Negative data generation**: How negative data is generated (label corruption, masking, etc.) is a critical design choice. Not shown.
- **No backpropagation**: The whole point of FF is to avoid backprop. The diagram should explicitly note this.

### Wrong Connections
- **NCA coupling**: The coupling between Forward-Forward goodness and NCA (Neural Cellular Automata) energy landscapes is a novel T4DM design, not part of the Forward-Forward algorithm. The claim "Goodness = -Energy" is an interesting theoretical alignment (both are scalar functions of state), but this equivalence is asserted without justification. In Hopfield networks, energy decreases as the network settles; in FF, goodness increases for positive data. These are related but the bidirectional coupling shown (FF_GOOD <--> NCA_EN) implies they are the same computation, which they are not.
- **NCA basin to FF positive phase feedback**: The arrow NCA_BASIN to NCA2FF to FF_POS implies the energy landscape drives FF learning. This reverses the actual intended direction: FF learning should *shape* the energy landscape, not the other way around. If the energy landscape drives FF, you have a circular dependency that may not converge.

### Recommendations
1. Show FF as multi-layer with per-layer goodness
2. Show negative data generation mechanism
3. Clearly separate FF (established) from NCA coupling (speculative)
4. Reconsider the bidirectional coupling: specify which direction is primary
5. Add explicit note: "No backpropagation required"

---

## 13. three_factor_rule.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/three_factor_rule.mermaid`

### Correct Wiring
- Three factors: pre-synaptic activity, post-synaptic activity, neuromodulator: correct (Fremaux & Gerstner, 2016)
- Hebbian term = Pre x Post: correct
- Final weight update = eta x Pre x Post x M: correct formulation
- DA as RPE gating LTP/LTD: correct (Reynolds & Wickens, 2002)
- ACh gating encoding vs retrieval mode: correct

### Missing Connections
- **Eligibility trace as intermediate**: The three-factor rule in its modern formulation includes an eligibility trace that bridges the temporal gap between Hebbian coincidence and neuromodulator arrival. The diagram jumps directly from Pre x Post to modulated weight update without showing the eligibility trace as the necessary temporal bridge. (This is shown in the separate eligibility_trace.mermaid, but the three-factor diagram should at least reference it.)
- **STDP timing dependence**: The Hebbian term "Pre x Post" does not capture spike-timing dependence. STDP (pre-before-post = LTP, post-before-pre = LTD) is a crucial refinement. The diagram treats it as rate-based Hebbian, not timing-based.
- **Inhibitory synapses**: The diagram only shows excitatory plasticity. GABAergic synapses also undergo plasticity with different rules (inhibitory STDP, Vogels et al., 2011).

### Wrong Connections
- **"Low DA + Pre x Post > 0 leads to LTD or none"**: This is not quite right. Low DA (DA dip / negative RPE) combined with an active eligibility trace leads to LTD. But if DA is simply at baseline (not dipped), the synapses should not change. The diagram conflates "low DA" with "negative RPE," which are different states.

### Recommendations
1. Add eligibility trace as an intermediate between Hebbian coincidence and neuromodulator gating
2. Distinguish rate-based Hebbian from STDP timing-dependent version
3. Fix "Low DA" to "Negative RPE (DA dip below baseline)"
4. Note that baseline DA = no change, not LTD

---

## 14. eligibility_trace.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/eligibility_trace.mermaid`

### Correct Wiring
- Pre and post spike pairing creates eligibility trace: correct
- Trace decays exponentially: correct
- DA arrives delayed (0.5-2s after event): correct timing (Izhikevich, 2007)
- Weight update = eta x DA x e(t): correct three-factor formulation
- Timeline showing temporal dynamics (pairing, decay, DA arrival): correct and helpful

### Missing Connections
- **Multiple neuromodulators**: The diagram only shows DA as the modulating signal. Other neuromodulators (NE, ACh, 5-HT) also interact with eligibility traces. NE can enhance trace formation; ACh can gate whether traces are formed at all.
- **Trace resetting**: After a successful credit assignment (DA arrival + active trace), the trace should be consumed/reset. Not shown.
- **Negative eligibility traces**: For anti-Hebbian learning or LTD, there should be negative traces (post-before-pre). Only positive traces are shown.

### Wrong Connections
- **Decay formula uses gamma and lambda**: The formula e(t+1) = gamma x lambda x e(t) mixes RL parameters (gamma = discount, lambda = trace decay) with the biological eligibility trace concept. In biology, the trace decay is simply exponential with a single time constant tau. The TD-lambda formulation is an RL abstraction, not a biological mechanism. This conflation is not wrong for a software system but is misleading if claiming biological plausibility.

### Recommendations
1. Add other neuromodulator interactions (NE enhances, ACh gates)
2. Add trace consumption/reset after credit assignment
3. Either use biological notation (single tau decay) or RL notation (gamma, lambda) but note which is which
4. Add negative eligibility traces

---

## 15. credit_assignment_flow.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/credit_assignment_flow.mermaid`

### Correct Wiring
- TD error (delta = r + gamma*V(s') - V(s)): correct Sutton & Barto formulation
- Positive TD error maps to DA burst, negative to DA pause: correct (Schultz et al., 1997)
- Eligibility trace evolution over time: correct
- 5-HT modulating lambda (temporal credit window): correct, matches Doya (2002)
- NE gating eligibility trace: plausible (arousal gating learning)

### Missing Connections
- **Successor representation**: Modern computational neuroscience suggests hippocampal place cells implement a successor representation (Stachenfeld et al., 2017), which provides an alternative credit assignment mechanism beyond eligibility traces. Worth noting as a research direction.
- **Actor-critic architecture**: The diagram shows TD learning but does not distinguish between the critic (value estimation, which maps to striatal DA prediction) and the actor (policy, which maps to dorsal striatum). This distinction matters for understanding which synapses the eligibility traces apply to.

### Wrong Connections
- **5-HT modulating lambda as "lambda' = lambda * (1 + 5-HT)"**: This formula means higher 5-HT always increases lambda (longer traces). But 5-HT's effect on temporal discounting is more nuanced: high 5-HT increases *patience* (willingness to wait for delayed reward), which maps to higher gamma (discount factor), not necessarily higher lambda (trace decay). Lambda and gamma have different functional roles. The diagram conflates them.
- **ACh modulating initial eligibility (e_init = ACh level)**: This implies ACh sets the starting magnitude of the eligibility trace. In biology, ACh gates *plasticity* at the synapse level (via muscarinic receptors modulating NMDA currents), not the magnitude of the initial trace. The effect is more like a binary gate than a scalar multiplier.

### Recommendations
1. Distinguish gamma (patience/discount) from lambda (trace decay) in 5-HT effects
2. Model ACh as a plasticity gate (binary-ish) rather than trace magnitude scalar
3. Consider adding actor-critic distinction
4. Note that this maps to dorsal and ventral striatal circuits

---

## 16. consolidation_stages.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/consolidation_stages.mermaid`

### Correct Wiring
- Two-stage model: synaptic consolidation (hours) then systems consolidation (days-years): correct (Dudai, 2004)
- LTP induction (NMDA-dependent): correct
- Protein synthesis window (2-6 hours): correct
- Synaptic tagging and capture: correct (Frey & Morris, 1997)
- NREM slow-wave sleep with spindles and SWR replay: correct
- HPC to neocortex transfer over time: correct (Frankland & Bontempi, 2005)
- Schema integration: correct (Tse et al., 2007)
- Interleaved replay of old and new: correct, critical for avoiding catastrophic interference
- Synaptic downscaling (homeostatic): correct (Tononi & Cirelli, 2006, SHY hypothesis)
- Glymphatic clearance: correct

### Missing Connections
- **Reconsolidation**: When a consolidated memory is retrieved, it enters a labile state requiring reconsolidation (Nader et al., 2000). This is absent. It matters because reconsolidation is when memories can be updated or distorted.
- **Multiple trace theory**: An alternative to full HPC independence is that episodic memories always retain a hippocampal trace (Nadel & Moscovitch, 1997). The diagram assumes full independence, which is the standard complementary learning systems (CLS) view but not the only model.
- **REM sleep role**: Only NREM is shown for sleep consolidation. REM sleep plays a role in emotional memory consolidation and creative integration (Walker & van der Helm, 2009). Missing.
- **Cortisol/stress modulation**: Stress hormones modulate consolidation. High cortisol during encoding enhances consolidation but impairs retrieval. Not shown (understandable for scope).

### Wrong Connections
- **Linear flow from NREM sleep to systems consolidation**: The diagram implies a strict linear pipeline: synaptic consolidation then sleep then systems consolidation. In reality, systems consolidation occurs over many sleep cycles spanning days to months. A single sleep cycle does not complete systems consolidation; it advances it incrementally.

### Recommendations
1. Add reconsolidation loop (retrieval makes memory labile again)
2. Add REM sleep phase for emotional/creative integration
3. Indicate that systems consolidation is iterative over many sleep cycles
4. Consider noting multiple trace theory as alternative to full HPC independence

---

## 17. memory_lifecycle.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/memory_lifecycle.mermaid`

### Correct Wiring
- Tripartite memory (episodic, semantic, procedural): correct taxonomy (Tulving, 1972)
- Short-term buffer with capacity limit: correct
- SWR-driven consolidation: correct
- Episodic to semantic extraction: correct
- Trace decay (exponential): correct
- Interference (similarity-based): correct
- Compression/abstraction as forgetting mechanism: correct, this is constructive forgetting
- Reconsolidation via retrieval feedback: shown (RESULT to REP), good

### Missing Connections
- **Emotional modulation of encoding**: The encoding pipeline (EXP to ENC to EMB) has no emotional or salience gating. In biology, amygdala activation strongly modulates hippocampal encoding strength. High-emotion events are encoded more strongly.
- **Context-dependent retrieval**: Retrieval depends on context overlap with encoding (encoding specificity principle, Tulving & Thomson, 1973). The retrieval system shows vector similarity but not explicit context matching.
- **Priming**: Repeated exposure to related content should lower retrieval thresholds (priming/repetition suppression). Not modeled.
- **Procedural memory formation**: The diagram shows procedural memory receiving from consolidation, but procedural memory (skills, habits) forms through a different pathway than episodic/semantic: it involves basal ganglia and cerebellar circuits, not hippocampal consolidation. Routing through the same SWR consolidation pipeline is incorrect for procedural learning.

### Wrong Connections
- **Procedural memory through SWR consolidation**: As noted above, procedural/habit learning is basal ganglia-dependent, not hippocampal. The diagram routes all memory types through the same hippocampal consolidation pathway. Procedural learning should have its own pathway through striatal reinforcement.

### Recommendations
1. Add emotional/amygdala modulation at encoding stage
2. Separate procedural memory formation (striatal pathway) from episodic/semantic (hippocampal pathway)
3. Add context encoding at storage and context matching at retrieval

---

## 18. nca_module_map.mermaid

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/nca_module_map.mermaid`

### Correct Wiring
- HPC internal: DG to CA3 to CA1 trisynaptic: correct
- SPC (spatial cells) to DG: correct (grid cells from EC to DG)
- Astrocyte tripartite synapse: correct concept (Perea et al., 2009)
- SWR coupling to CA3: correct
- Sleep spindles to CA1: plausible (spindles affect CA1 through thalamocortical input)
- Adenosine dynamics driving sleep state: correct
- Glymphatic to astrocyte connection: correct

### Missing Connections
- **EC entirely missing**: The hippocampal system has no entorhinal cortex input or output. DG just receives from SPC. The trisynaptic circuit is incomplete without EC.
- **CA1 output pathway**: CA1 connects to energy landscape (EN) but in biology CA1 projects to subiculum and EC. There is no biological output from hippocampus directly to an "energy landscape."
- **Neuromodulator targets**: VTA/DA goes to coupling (CP), Raphe goes to energy (EN), LC goes to attractor (AT). These are software mappings, not biological projections. In biology: VTA projects to HPC, PFC, NAc; Raphe projects broadly; LC projects even more broadly. The diagram maps neuromodulators to NCA software components, which is fine for the software architecture but should not be interpreted as biological wiring.
- **Striatal MSN**: StriatalMSN is in the neuromodulatory systems but has no clear biological role in the diagram beyond connecting to coupling. In biology, MSNs are targets of DA, not neuromodulatory sources. They are miscategorized.

### Wrong Connections
- **StriatalMSN as a neuromodulatory system**: MSNs (medium spiny neurons) are the principal neurons of the striatum. They are *targets* of neuromodulators (especially DA via D1 and D2 receptors), not neuromodulator *sources*. Placing them alongside VTA, Raphe, and LC as "Neuromodulatory Systems" is a category error.
- **CAP (capsules) to FF (forward-forward)**: Labeled as "Capsule-NCA Coupling." This is a software design decision, not a biological connection. Capsule networks and the Forward-Forward algorithm are not biologically coupled systems.
- **TG (theta-gamma) to CA3**: Theta-gamma coupling is a hippocampal oscillatory phenomenon, not a module that *sends to* CA3. Theta-gamma coupling *occurs in* CA3/CA1 and is modulated by the medial septum. Making it a separate module that feeds into CA3 reverses the relationship.

### Recommendations
1. Add EC as input/output for hippocampal system
2. Move StriatalMSN from "Neuromodulatory Systems" to a "Basal Ganglia" or "Target Regions" subgraph
3. Move ThetaGamma from "Learning Mechanisms" into the hippocampal system as an oscillatory property, not a separate input
4. Clearly distinguish biological wiring from software architecture mapping

---

## 19. 02_bioinspired_components.mmd

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/02_bioinspired_components.mmd`

### Correct Wiring
- DG sparse encoder with k-winner-take-all (top 2%): correct sparsity level (Rolls, 2013)
- Lateral inhibition in sparse encoding: correct
- Two-compartment neuron (basal/apical): correct, matches Larkum (2013) dendritic computation model
- Apical compartment receiving context/feedback: correct
- Mismatch signal from soma: correct (predictive processing)
- Attractor network (CA3) with Hopfield/Hebbian weights and energy function: correct
- Eligibility traces with fast (5s) and slow (60s) timescales: biologically plausible
- Fast episodic store with one-shot learning and salience-based eviction: good computational analog to hippocampal fast encoding
- Salience = DA x NE x ACh: plausible multiplicative interaction

### Missing Connections
- **Dendritic compartment interaction with attractor network**: The dendritic processing and attractor network are shown as separate components. In biology, CA3 pyramidal cells *are* the dendritic neurons, and their recurrent connections form the attractor. These should be integrated.
- **Inhibitory interneurons**: Pattern separation in DG depends on inhibitory interneurons, not just lateral inhibition as a process. Show at least one interneuron population.

### Wrong Connections
- No major wrong connections. This diagram is primarily a software component map with biological inspiration labels, and it does a reasonable job at that level.

### Recommendations
1. Integrate dendritic processing into the attractor network (CA3 cells have dendrites)
2. Add explicit interneuron population for DG pattern separation

---

## 20. 06_memory_systems.mmd

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/06_memory_systems.mmd`

### Assessment
This is primarily a software architecture diagram. The biological mapping is: Fast Episodic Store = hippocampal fast encoding, consolidation pipeline = sleep replay, episodic/semantic/procedural = Tulving's taxonomy. The architecture is reasonable from a CLS (complementary learning systems) perspective.

### Missing Biological Elements
- **No neuromodulatory gating of encoding**: The fast episodic store mentions salience = DA x NE x ACh but the main memory systems diagram does not show neuromodulatory control.
- **No emotional memory pathway**: Amygdala-dependent emotional encoding is absent.
- **Procedural memory should bypass hippocampal consolidation**: As noted in the lifecycle diagram.

### Recommendations
1. Add neuromodulatory gating at the Working Memory to Fast Episodic transition
2. Route procedural memory through a separate (striatal) pathway

---

## 21. 07_class_bioinspired.mmd

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/07_class_bioinspired.mmd`

### Assessment
Class diagram for software components. Not a biological circuit diagram per se. The class hierarchy is reasonable: SparseEncoder with AdaptiveSparseEncoder, AttractorNetwork with ModernHopfieldNetwork, EligibilityTrace with LayeredEligibilityTrace.

### Biological Plausibility Notes
- ModernHopfieldNetwork inheriting from AttractorNetwork: conceptually sound, Ramsauer et al. (2021)
- LayeredEligibilityTrace with fast and slow timescales: biologically plausible dual-timescale traces
- DendriticNeuron with basal/apical compartments and coupling parameter: correct abstraction of Larkum model
- FastEpisodicStore with salience-based eviction: reasonable hippocampal analog

### Recommendations
- No major biological wiring issues (this is a class diagram, not a circuit diagram)

---

## 22. 08_consolidation_pipeline.mmd

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/08_consolidation_pipeline.mmd`

### Correct Wiring
- Multiple trigger types (time-based, threshold, sleep, manual): reasonable
- FES consolidation based on replay count, salience, age: plausible
- Episodic consolidation with clustering and prototype extraction: matches CLS theory
- Semantic consolidation with entity extraction and knowledge integration: reasonable
- Sleep consolidation with light sleep, deep sleep, REM: correct phase progression

### Missing Connections
- **Neuromodulatory state changes during consolidation**: During sleep, neuromodulator levels change dramatically (ACh drops in NREM, rises in REM; NE drops in both NREM and REM). These changes are functionally critical for consolidation and are not represented.
- **Feedback from semantic to episodic**: When semantic knowledge is updated, it should feed back to episodic memories (schema-mediated encoding, Tse et al., 2007). The flow is shown as unidirectional (episodic to semantic) but consolidation involves bidirectional transfer.

### Wrong Connections
- **Sleep consolidation to semantic and procedural as parallel outputs**: In biology, NREM preferentially consolidates declarative (episodic/semantic) memories, while REM preferentially consolidates procedural and emotional memories (Walker, 2009). The diagram treats them as receiving equally from sleep consolidation, missing this dissociation.

### Recommendations
1. Add neuromodulator state changes across sleep phases
2. Add NREM preferentially consolidating declarative memories and REM preferentially consolidating procedural/emotional
3. Add bidirectional semantic-episodic interaction during consolidation

---

## 23. 12_learning_subsystem.mmd

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/12_learning_subsystem.mmd`

### Assessment
Software learning architecture. The cycle Gate to Eligibility to Hebbian to Retrieval to Gate is a reasonable control loop.

### Biological Plausibility Notes
- Bayesian learning rate and Thompson sampling in the gate: not biologically plausible as stated, but reasonable as a computational abstraction of uncertainty-driven exploration.
- Hebbian learner with dw = eta x pre x post x DA: correct three-factor formulation
- Three trace timescales (5s, 20s, 60s): biologically plausible range
- Neural reranker for retrieval: no direct biological analog but functionally reasonable

### Recommendations
- Note which components are biologically inspired vs purely computational

---

## 24. 13_neuromodulation_subsystem.mmd

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/13_neuromodulation_subsystem.mmd`

### Correct Wiring
- Five systems (DA, NE, ACh, 5-HT, GABA) with bidirectional communication to orchestra coordinator: reasonable architecture
- DA: RPE, prediction model, burst/dip: correct
- NE: arousal detection, novelty tracking, exploration rate: correct
- ACh: encoding/retrieval mode, plasticity gate: correct
- 5-HT: temporal discount, eligibility modification, mood state: correct
- GABA: lateral inhibition, WTA, sparsity control: correct

### Missing Connections
- **Cross-modulator interactions**: The diagram only shows each system communicating with the central orchestra. In biology, neuromodulator systems interact directly: 5-HT inhibits DA (raphe to VTA), NE excites ACh (LC to BF), etc. These lateral interactions are missing.
- **Glutamate**: Glutamate is the primary excitatory neurotransmitter and is not represented. While it is not a "neuromodulator" in the classical sense, its interactions with GABA and the neuromodulators are critical.

### Recommendations
1. Add direct cross-modulator interactions (not just through orchestra)
2. Consider adding glutamate as a sixth system

---

## 25. 24_class_neuromod.mmd

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/24_class_neuromod.mmd`

### Assessment
Well-structured class hierarchy. NeuromodulatorSystem as abstract base with five concrete implementations. UnifiedState as a composite. The API design (get_level, set_level, update, decay_to_baseline) is clean.

### Biological Plausibility Notes
- GABA inheriting from NeuromodulatorSystem: GABA is a neurotransmitter, not a neuromodulator in the strict sense. It operates at the synaptic level, not through volume transmission like DA/NE/5-HT/ACh. Treating it the same way is a simplification.
- ArousalState with four levels (DROWSY, ALERT, VIGILANT, HYPERAROUSED): maps to Yerkes-Dodson inverted-U curve, which is correct.
- ACHMode with three states (ENCODING, RETRIEVAL, BALANCED): correct per Hasselmo model, though in biology it is a continuum, not discrete states.

### Recommendations
- Consider making GABA a separate class type (InhibitorySystem) rather than a NeuromodulatorSystem

---

## 26. 32_state_consolidation.mmd

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/32_state_consolidation.mmd`

### Correct Wiring
- AWAKE to NREM_SLEEP to REM_SLEEP to PRUNING cycle: correct basic structure
- NREM: SWR replay with entity extraction and semantic transfer: correct
- REM: HDBSCAN clustering and LLM-based abstraction: creative computational interpretation
- NREM 75%, REM 25% of cycle: correct approximate proportions
- Pruning with homeostatic plasticity targeting 3% activity: plausible (synaptic homeostasis hypothesis)
- Preserving tagged synapses during pruning: correct (synaptic tagging, Frey & Morris, 1997)
- 10-20x temporal compression in SWR: correct

### Missing Connections
- **Multiple NREM-REM cycles**: The diagram shows a single AWAKE to NREM to REM to PRUNING to AWAKE cycle. In biological sleep, there are typically 4-5 NREM-REM cycles per night, with varying proportions (early night NREM-heavy, late night REM-heavy). The single-cycle model is a simplification.
- **Dreaming/REM content**: REM is shown as clustering/abstraction, but REM also involves emotional memory processing and creative recombination. Only the abstraction aspect is modeled.

### Wrong Connections
- **Pruning as a separate phase after REM**: Synaptic downscaling (the SHY hypothesis) occurs primarily during NREM slow-wave sleep, not as a separate post-REM phase. The up-states and down-states of slow oscillations naturally implement synaptic downscaling. Placing pruning after REM reverses the biological ordering.

### Recommendations
1. Move pruning/downscaling into the NREM phase (during slow oscillations)
2. Consider multiple NREM-REM cycles
3. Add emotional processing to REM phase

---

## 27. 33_state_neuromod.mmd

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/33_state_neuromod.mmd`

### Correct Wiring
- DA baseline, burst (positive RPE), dip (negative RPE): correct
- DA burst decay tau=2s, dip decay tau=5s: plausible timescales (burst is faster than dip recovery)
- Arousal states (DROWSY to ALERT to VIGILANT to HYPERAROUSED): maps to Yerkes-Dodson
- Optimal performance at ALERT (NE 0.4-0.6): correct
- Impaired cognition at HYPERAROUSED (NE > 1.5): correct
- ACh encoding (>0.6, high plasticity, LTP) vs retrieval (<0.4, low plasticity, pattern completion): correct per Hasselmo

### Missing Connections
- **Cross-state interactions**: When DA bursts, it should also transiently affect NE and ACh states. The diagram shows each neuromodulator system independently transitioning, but they interact. For example, a strong DA burst (unexpected reward) should also increase NE (arousal) and ACh (encoding mode for the new information).
- **Serotonin states**: 5-HT is mentioned in the baseline but has no dedicated state diagram. It should have states like "patient" (high 5-HT, waiting for delayed reward) and "impulsive" (low 5-HT).
- **Sleep state**: During sleep, all neuromodulator levels change dramatically (NE and 5-HT drop to near zero, ACh drops in NREM but rises in REM). No sleep-related neuromodulator states are shown.

### Recommendations
1. Add cross-modulator state interactions
2. Add serotonin state diagram
3. Add sleep-related neuromodulator state transitions

---

## 28. 34_state_memory_gate.mmd

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/34_state_memory_gate.mmd`

### Assessment
This is a software state machine for a Bayesian memory gate. It is not a biological circuit diagram. The Thompson sampling and Bayesian learning rate approach is a reasonable computational abstraction of uncertainty-driven memory gating.

### Biological Analog
The closest biological analog is the hippocampal novelty/familiarity detection circuit in CA1 (comparing EC input with CA3 retrieved patterns) combined with neuromodulatory gating (DA modulates whether LTP occurs). The diagram captures the *function* but not the *mechanism*.

### Recommendations
- Add a note mapping this to the biological analog (CA1 comparator + DA gating)

---

## 29. 43_seq_consolidation.mmd

**File**: `/mnt/projects/t4d/t4dm/docs/diagrams/43_seq_consolidation.mmd`

### Correct Wiring
- NREM phase (75%): SWR replay with similarity-based sequence building, temporal compression, entity extraction: correct computational analog
- REM phase (25%): HDBSCAN clustering with LLM abstraction: creative but not biologically grounded
- Pruning: synaptic downscaling with preservation of tagged connections: correct concept
- Interleaving of old and new memories in replay: implied by sequence building

### Missing Connections
- **Spindle-SWR coupling in the sequence**: The consolidation sequence jumps from candidate selection to SWR replay but does not include the spindle-SWR temporal coupling that is critical for cortical plasticity during consolidation.
- **Neuromodulator state during sleep**: The sequence does not show the neuromodulator changes that occur during sleep (ACh drop enabling replay, NE drop enabling plasticity).
- **REM rebound**: If NREM is interrupted, there should be REM rebound. No error handling for interrupted consolidation.

### Wrong Connections
- **Pruning to activity target of 3%**: The 3% target is stated as a fixed parameter. In biology, the homeostatic set point for synaptic strength is regulated dynamically based on recent history, not a fixed percentage. This is a software simplification that could lead to pathological behavior if the natural activity level should be different.

### Recommendations
1. Add spindle-mediated cortical plasticity step between replay and semantic transfer
2. Add neuromodulator state annotations (ACh low, NE low during NREM)
3. Make the pruning target adaptive rather than fixed at 3%

---

## Cross-Cutting Issues

### Issue 1: Inconsistent Hippocampal Circuit Across Diagrams

The hippocampal circuit is represented differently across `hippocampal_circuit.mermaid`, `hpc_trisynaptic.mermaid`, `nca_module_map.mermaid`, and `02_bioinspired_components.mmd`. The trisynaptic diagram is the most accurate, with EC Layer II/III distinction, subiculum, and proper output chain. The others should be updated to match.

### Issue 2: Systematic Absence of Inhibitory Interneurons

Across all diagrams, GABAergic inhibitory interneurons are dramatically underrepresented. In the brain, approximately 20% of cortical neurons are inhibitory, and they are critical for:
- Pattern separation (DG basket cells)
- Ripple generation (CA1 PV+ interneurons)
- Gain control (all regions)
- Oscillation generation (all rhythms depend on E/I balance)

The diagrams treat inhibition as a process label ("lateral inhibition", "winner-take-all") rather than as explicit circuit elements.

### Issue 3: Missing PFC Feedback to Neuromodulator Sources

No diagram shows prefrontal cortex feedback to VTA, LC, DRN, or BF. This top-down control is critical for cognitive regulation of neuromodulation (e.g., goal-directed attention modulating LC-NE, emotion regulation modulating VTA-DA). It appears only as an input in the VTA circuit but not in the broader system diagrams.

### Issue 4: Procedural Memory Routing Error

Multiple diagrams (memory_lifecycle, 06_memory_systems, 08_consolidation_pipeline) route procedural memory through hippocampal consolidation. Procedural/habit learning is basal ganglia-dependent (dorsal striatum) and cerebellar, operating through reinforcement learning rather than hippocampal replay. This is a systematic architectural error.

### Issue 5: Forward-Forward and Capsule Biological Claims

The FF and capsule diagrams add neuromodulator integration as if it were part of these algorithms. These are novel T4DM extensions and should be clearly distinguished from the established algorithms. Conflating them risks misrepresenting what Hinton's Forward-Forward (2022) and Sabour, Fross & Hinton (2017) capsule networks actually specify.

---

## Priority Fixes

### Critical (Incorrect Biology)
1. Fix SWR frequency in sleep_cycle.mermaid: change "SWR Events (0.5-2 Hz)" to "Slow Oscillations (0.5-1 Hz)"
2. Remove SIGMA to CA3 arrow in spindle_ripple_coupling.mermaid (spindles do not trigger SWRs)
3. Move StriatalMSN out of "Neuromodulatory Systems" in nca_module_map.mermaid
4. Fix grid cell to place cell direction in hippocampal_circuit.mermaid
5. Separate procedural memory from hippocampal consolidation pathway

### Important (Missing Critical Elements)
6. Add EC as input/output for hippocampal circuits missing it
7. Add subiculum in circuits that skip it
8. Add PFC feedback to neuromodulator sources
9. Add inhibitory interneuron populations (at least in hippocampal and pattern separation circuits)
10. Add 5-HT inhibition of VTA DA neurons

### Nice to Have (Refinements)
11. Add reconsolidation loops
12. Add D2 autoreceptor feedback in VTA circuit
13. Add cross-neuromodulator direct interactions
14. Add sleep-state neuromodulator level changes
15. Distinguish T4DM extensions from established algorithms in FF and capsule diagrams

---

## References

- Amaral & Witter (1989). The three-dimensional organization of the hippocampal formation. Neuroscience.
- Aston-Jones & Cohen (2005). An integrative theory of locus coeruleus-norepinephrine function. Annual Review of Neuroscience.
- Borbely (1982). A two process model of sleep regulation. Human Neurobiology.
- Buzsaki (2015). Hippocampal sharp wave-ripple. Physiological Reviews.
- Daw et al. (2002). Opponent interactions between serotonin and dopamine. Neural Networks.
- Doya (2002). Metalearning and neuromodulation. Neural Networks.
- Dudai (2004). The neurobiology of consolidations. Annual Review of Psychology.
- Foster & Wilson (2006). Reverse replay of behavioural sequences in hippocampal place cells. Nature.
- Frankland & Bontempi (2005). The organization of recent and remote memories. Nature Reviews Neuroscience.
- Fremaux & Gerstner (2016). Neuromodulated spike-timing-dependent plasticity. Frontiers in Computational Neuroscience.
- Frey & Morris (1997). Synaptic tagging and long-term potentiation. Nature.
- Grace & Bunney (1984). The control of firing pattern in nigral dopamine neurons. Journal of Neuroscience.
- Hablitz et al. (2020). Circadian control of brain glymphatic and lymphatic fluid flow. Nature Communications.
- Halassa et al. (2009). Astrocytic modulation of sleep homeostasis. Neuron.
- Harley (2007). Norepinephrine and the dentate gyrus. Progress in Brain Research.
- Hasselmo (1999). Neuromodulation: acetylcholine and memory consolidation. Trends in Cognitive Sciences.
- Hasselmo (2006). The role of acetylcholine in learning and memory. Current Opinion in Neurobiology.
- Hasselmo & Stern (2006). Mechanisms underlying working memory for novel information. Trends in Cognitive Sciences.
- Henze et al. (2000). Single granule cells reliably discharge targets in the hippocampal CA3 network in vivo. Nature Neuroscience.
- Hinton (2022). The Forward-Forward Algorithm. arXiv:2212.13345.
- Izhikevich (2007). Solving the distal reward problem through linkage of STDP and dopamine signaling. Cerebral Cortex.
- Jhou et al. (2009). The rostromedial tegmental nucleus. Journal of Neuroscience.
- Kudrimoti et al. (1999). Reactivation of hippocampal cell assemblies. Journal of Neuroscience.
- Lansink et al. (2009). Hippocampus leads ventral striatum in replay. PLoS Biology.
- Larkum (2013). A cellular mechanism for cortical associations. Neuron.
- Lee & Wilson (2002). Memory of sequential experience in the hippocampus during slow wave sleep. Neuron.
- Lisman & Grace (2005). The hippocampal-VTA loop. Neuron.
- McClelland et al. (1995). Why there are complementary learning systems. Psychological Review.
- Miyazaki et al. (2014). Serotonergic projections to the amygdala facilitate fear learning. Nature Neuroscience.
- Morales & Margolis (2017). Ventral tegmental area: cellular heterogeneity. Nature Reviews Neuroscience.
- Nadel & Moscovitch (1997). Memory consolidation, retrograde amnesia and the hippocampal complex. Current Opinion in Neurobiology.
- Nader et al. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation. Nature.
- Ngo et al. (2013). Auditory closed-loop stimulation of the sleep slow oscillation. Neuron.
- Perea et al. (2009). Tripartite synapses: astrocytes process and control synaptic information. Trends in Neurosciences.
- Ramsauer et al. (2021). Hopfield networks is all you need. ICLR.
- Reynolds & Wickens (2002). Dopamine-dependent plasticity of corticostriatal synapses. Neural Networks.
- Rolls (2013). The mechanisms for pattern completion and pattern separation in the hippocampus. Frontiers in Systems Neuroscience.
- Sabour, Fross & Hinton (2017). Dynamic routing between capsules. NeurIPS.
- Saper et al. (2001). The sleep switch: hypothalamic control of sleep and wakefulness. Trends in Neurosciences.
- Saper et al. (2005). Hypothalamic regulation of sleep and circadian rhythms. Nature.
- Scammell et al. (2001). An adenosine A2a agonist increases sleep. Neuroscience.
- Schultz et al. (1997). A neural substrate of prediction and reward. Science.
- Stachenfeld et al. (2017). The hippocampus as a predictive map. Nature Neuroscience.
- Staresina et al. (2015). Hierarchical nesting of slow oscillations, spindles and ripples. Nature Neuroscience.
- Steriade et al. (1993). A novel slow oscillation of neocortical neurons in vivo. Journal of Neuroscience.
- Tononi & Cirelli (2006). Sleep function and synaptic homeostasis. Sleep Medicine Reviews.
- Tse et al. (2007). Schemas and memory consolidation. Science.
- Tulving (1972). Episodic and semantic memory. In Organization of Memory.
- Tulving & Thomson (1973). Encoding specificity and retrieval processes. Psychological Review.
- Vogels et al. (2011). Inhibitory plasticity balances excitation and inhibition. Science.
- Walker (2009). The role of sleep in cognition and emotion. Annals of the New York Academy of Sciences.
- Walker & van der Helm (2009). Overnight therapy? The role of sleep in emotional brain processing. Psychological Bulletin.
- Xie et al. (2013). Sleep drives metabolite clearance from the adult brain. Science.
- Yamaguchi et al. (2011). Glutamatergic neurons in the VTA. Neuron.
