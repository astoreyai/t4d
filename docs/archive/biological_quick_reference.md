# T4DM Biological Systems - Quick Reference

**Fast lookup for parameter validation and biological ranges**

---

## FIRING RATES (Hz)

| System | Tonic | Phasic | Biological Source |
|--------|-------|--------|-------------------|
| VTA Dopamine | 4.5 | 20-40 (burst), 0-2 (pause) | Schultz 1998 |
| Raphe 5-HT | 2.5 | 8.0 (max) | Hajos 2007 |
| LC Norepinephrine | 1-3 | 15 (peak) | Aston-Jones 2005 |
| Sharp-Wave Ripples | - | 150-250 (ripple) | Buzsáki 2015 |

---

## OSCILLATION FREQUENCIES (Hz)

| Band | Range | Peak | Biological Function |
|------|-------|------|---------------------|
| Delta | 0.5-4 | 1.5 | Slow-wave sleep, consolidation |
| Theta | 4-8 | 6.0 | Memory encoding, navigation |
| Alpha | 8-13 | 10.0 | Idling, inhibition |
| Beta | 13-30 | 20.0 | Motor control, cognition |
| Spindles | 11-16 | 13.0 | NREM stage 2, memory gate |
| Gamma (low) | 30-60 | 40.0 | Local processing |
| Gamma (high) | 60-100 | 80.0 | Attention binding |
| Ripples | 150-250 | 180 | Memory replay |

---

## TIME CONSTANTS

| Process | Value | Unit | Source |
|---------|-------|------|--------|
| AMPA EPSC | 2 | ms | Dingledine 1999 |
| NMDA EPSC | 50 | ms | Dingledine 1999 |
| ACh decay | 0.5 | s | AChE kinetics |
| GABA decay | 0.2 | s | GAT uptake |
| Glu decay | 0.1 | s | EAAT uptake |
| DA decay | 2.0 | s | MAO/COMT |
| 5-HT decay | 5.0 | s | SERT/MAO |
| NE decay | 1.5 | s | NET/MAO |
| SWR duration | 80 | ms | Buzsáki 2015 |
| Spindle duration | 500-2000 | ms | Steriade 1993 |
| Adenosine accumulation | 0.04 | /hr | Borbély 1982 |
| Caffeine half-life | 5.0 | hr | Pharmacology |

---

## NEUROTRANSMITTER INTERACTIONS

| Source | Target | Effect | Strength | Literature |
|--------|--------|--------|----------|------------|
| 5-HT | VTA DA | Inhibit | -0.3 | Di Matteo 2001 |
| NE | Alpha | Suppress | -0.4 | Sara 2009 |
| ACh | Theta | Enhance | +0.5 | Hasselmo 2005 |
| DA | Beta | Enhance | +0.3 | Gerfen 2011 |
| Adenosine | DA | Suppress | -0.3 | Basheer 2004 |
| Adenosine | NE | Suppress | -0.4 | Sleep literature |
| ACh (high) | SWR | Block | Threshold 0.3 | Vandecasteele 2014 |
| NE (low) | Glymphatic | Enable | < 0.2 | Xie 2013 |

---

## HIPPOCAMPAL PARAMETERS

| Component | Parameter | Value | Biological |
|-----------|-----------|-------|------------|
| DG cells | Dimension | 4096 | ~10^6 biological |
| DG sparsity | Activation | 4% | 0.5% biological |
| DG separation | Threshold | 0.55 | Similarity cutoff |
| CA3 cells | Dimension | 1024 | ~3×10^5 biological |
| CA3 Hopfield beta | Temperature | 8.0 | Retrieval sharpness |
| CA3 capacity | Patterns | 1000 | Modern Hopfield |
| CA1 novelty | Threshold | 0.3 | EC-CA3 mismatch |
| Place cells | Count | 100 | Simplified |
| Place field sigma | Width | 0.15 | Gaussian RF |
| Grid scales | Modules | 3 | (0.3, 0.5, 0.8) |

---

## SLEEP PARAMETERS

| System | Parameter | Value | Source |
|--------|-----------|-------|--------|
| Adenosine baseline | Rested | 0.1 | Borbély 1982 |
| Adenosine max | Exhausted | 1.0 | Normalized |
| Sleep onset | Threshold | 0.7 | Process S |
| Wake threshold | Cleared | 0.2 | Can wake |
| Clearance (deep NREM) | Rate | 0.7 (70%) | Xie 2013 |
| Clearance (wake) | Rate | 0.3 (30%) | Xie 2013 |
| Clearance ratio | Sleep/Wake | 2.0x | Xie 2013 |
| SWR (NREM deep) | Probability | 0.9 | High |
| SWR (quiet wake) | Probability | 0.3 | Low |
| SWR (REM) | Probability | 0.0 | Blocked (ACh) |
| Spindle density | NREM2 | 2-5/min | EEG literature |
| Delta up-state | Duration | 200-500 ms | Steriade 1993 |
| Delta down-state | Duration | 200-500 ms | Steriade 1993 |

---

## DIFFUSION CONSTANTS

| NT | D (diffusion) | Range | Notes |
|----|---------------|-------|-------|
| Dopamine | 0.02 | Slow | Volume transmission |
| Serotonin | 0.015 | Very slow | Widespread |
| Norepinephrine | 0.03 | Moderate | LC projections |
| Acetylcholine | 0.025 | Moderate | Cholinergic |
| GABA | 0.05 | Fast | Local inhibition |
| Glutamate | 0.08 | Fastest | Synaptic |

---

## STRIATAL PARAMETERS

| Cell Type | DA Effect | Strength | Pathway |
|-----------|-----------|----------|---------|
| D1-MSN | Excite | +0.7 | Direct (GO) |
| D2-MSN | Inhibit | -0.5 | Indirect (NO-GO) |
| Up-state threshold | Voltage | -50 mV | Bistable |
| Down-state threshold | Voltage | -80 mV | Resting |

---

## NEUROMODULATOR FUNCTIONS

| NT | Primary Role | Secondary Role | Brain Region |
|----|--------------|----------------|--------------|
| Dopamine | Reward prediction error | Motivation, action | VTA, SNc |
| Serotonin | Patience, temporal discounting | Mood, impulse control | Raphe nuclei |
| Norepinephrine | Surprise, arousal | Signal-to-noise, reset | Locus coeruleus |
| Acetylcholine | Encoding vs retrieval | Attention, theta | Basal forebrain, MS |

---

## COMPUTATIONAL FRAMEWORKS

| Framework | Implementation | Biological Basis |
|-----------|----------------|------------------|
| Temporal Difference Learning | VTA dopamine RPE | Schultz 1997 |
| Opponent Process | Raphe-VTA inhibition | Solomon & Corbit 1974 |
| Adaptive Gain Theory | LC phasic/tonic modes | Aston-Jones & Cohen 2005 |
| Two-Process Model | Adenosine + circadian | Borbély 1982 |
| Modern Hopfield | CA3 autoassociation | Ramsauer et al. 2020 |
| Theta-Gamma Code | PAC in hippocampus | Lisman & Jensen 2013 |
| Glymphatic System | Sleep waste clearance | Nedergaard 2013 |

---

## VALIDATION QUICK CHECK

**Frequency Ranges** (all must be within bounds):
```python
assert 4.0 <= theta_freq <= 8.0
assert 8.0 <= alpha_freq <= 13.0
assert 13.0 <= beta_freq <= 30.0
assert 30.0 <= gamma_freq <= 100.0
assert 0.5 <= delta_freq <= 4.0
assert 11.0 <= spindle_freq <= 16.0
assert 150.0 <= ripple_freq <= 250.0
```

**Firing Rates**:
```python
assert 4.0 <= vta_tonic <= 5.0
assert 2.0 <= raphe_tonic <= 3.0
assert 1.0 <= lc_tonic <= 5.0
assert lc_phasic <= 20.0
```

**Sleep/Wake Gating**:
```python
assert ach_level < 0.3 for SWR  # High ACh blocks SWR
assert ne_level < 0.2 for glymphatic  # High NE blocks clearance
assert adenosine >= 0.7 for sleep_onset
assert adenosine <= 0.2 for wake_allowed
```

**Network Connectivity**:
```python
assert "vta" in neural_field.sources  # Dopamine injection
assert "raphe" in neural_field.sources  # Serotonin injection
assert "lc" in neural_field.sources  # NE injection
assert raphe.inhibits(vta)  # 5-HT → DA inhibition
assert ca1.projects_to(vta)  # Novelty signal
```

---

## COMMON BIOLOGICAL PITFALLS

1. **Ripple Frequency**: MUST be 150-250 Hz (not 100 Hz gamma!)
2. **Alpha-NE**: NEGATIVE relationship (NE suppresses alpha)
3. **ACh-SWR**: High ACh BLOCKS SWRs (REM has no replay)
4. **Glymphatic-NE**: Low NE required (astrocyte expansion)
5. **DG Sparsity**: ~0.5% biological, 4% practical compromise
6. **PAC Direction**: Theta PHASE → Gamma AMPLITUDE (not reverse)
7. **Delta Up-states**: Consolidation window (not down-states)
8. **Spindle-Delta**: Spindles IN up-states (100-200ms delay)

---

## FILE LOCATIONS

| Module | File Path |
|--------|-----------|
| VTA | `/mnt/projects/t4d/t4dm/src/t4dm/nca/vta.py` |
| Raphe | `/mnt/projects/t4d/t4dm/src/t4dm/nca/raphe.py` |
| LC | `/mnt/projects/t4d/t4dm/src/t4dm/nca/locus_coeruleus.py` |
| ACh | `/mnt/projects/t4d/t4dm/learning/acetylcholine.py` |
| Hippocampus | `/mnt/projects/t4d/t4dm/src/t4dm/nca/hippocampus.py` |
| Spatial Cells | `/mnt/projects/t4d/t4dm/src/t4dm/nca/spatial_cells.py` |
| Pattern Separation | `/mnt/projects/t4d/t4dm/src/t4dm/memory/pattern_separation.py` |
| Oscillators | `/mnt/projects/t4d/t4dm/src/t4dm/nca/oscillators.py` |
| SWR Coupling | `/mnt/projects/t4d/t4dm/src/t4dm/nca/swr_coupling.py` |
| Sleep Spindles | `/mnt/projects/t4d/t4dm/src/t4dm/nca/sleep_spindles.py` |
| Adenosine | `/mnt/projects/t4d/t4dm/src/t4dm/nca/adenosine.py` |
| Glymphatic | `/mnt/projects/t4d/t4dm/src/t4dm/nca/glymphatic.py` |
| Astrocyte | `/mnt/projects/t4d/t4dm/src/t4dm/nca/astrocyte.py` |
| Glutamate | `/mnt/projects/t4d/t4dm/src/t4dm/nca/glutamate_signaling.py` |
| Striatal MSN | `/mnt/projects/t4d/t4dm/src/t4dm/nca/striatal_msn.py` |

---

## CITATION QUICKLIST

**Most Cited** (>1000 citations):
- Schultz (1998): Dopamine RPE - *6,847 citations*
- Borbély (1982): Two-process model - *2,301 citations*
- Buzsáki (2002): Theta oscillations - *2,456 citations*
- Aston-Jones & Cohen (2005): LC adaptive gain - *1,891 citations*
- Xie et al. (2013): Glymphatic system - *1,821 citations*
- Dayan & Yu (2006): Uncertainty/NE - *1,456 citations*
- Lisman & Jensen (2013): Theta-gamma - *1,124 citations*

**Foundational**:
- Marr (1971): CA3 autoassociation
- O'Keefe & Dostrovsky (1971): Place cells
- Hafting et al. (2005): Grid cells (Nobel 2014)
- Steriade et al. (1993): Slow oscillations

**Recent Advances**:
- Ramsauer et al. (2020): Modern Hopfield networks
- Fultz et al. (2019): CSF oscillations
- Latchoumane et al. (2017): Spindle-memory coupling

---

**Last Updated**: 2026-01-04
**Version**: Phase 3 Complete
**Related Documents**:
- `biological_validation.md` (detailed validation)
- `biological_network_diagram.md` (visual maps)
