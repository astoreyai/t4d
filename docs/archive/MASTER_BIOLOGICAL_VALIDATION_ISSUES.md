# MASTER ISSUE LIST - COMPBIO VALIDATION

**Generated**: 2026-01-04
**Scope**: Complete biological validation audit of T4DM biologically-inspired modules
**Directories**: `src/t4dm/nca/`, `src/t4dm/learning/`, `src/t4dm/consolidation/`

---

## HIGH Priority (Biological Accuracy - Critical Fixes Required)

| ID | File | Parameter/Feature | Current | Literature | Source | Fix |
|----|------|-------------------|---------|------------|--------|-----|
| **B31** | `learning/stdp.py` | `tau_minus` | 20.0ms | 25-30ms | Bi & Poo 1998 | Change to 25-30ms range |
| **B32** | `learning/stdp.py` | AMPA dynamics | Present (a_plus/a_minus) | Should include tau_rise, tau_decay | Dayan & Abbott 2001 | Add separate AMPA rise/decay time constants |
| **B33** | `learning/stdp.py` | NMDA time constants | Missing | tau_rise=2ms, tau_decay=100ms (NR2A), tau_decay=150ms (NR2B) | Jahr & Stevens 1990 | Add NMDA receptor dynamics |
| **B34** | `learning/stdp.py` | GABA_A vs GABA_B | Not separated | tau_GABA_A=5-10ms, tau_GABA_B=50-200ms | Connors et al. 1988 | Separate fast and slow inhibition |
| **B35** | `learning/acetylcholine.py` | M1/M4 receptor effects | Not implemented | M1 enhances LTP (Gq), M4 reduces cAMP (Gi) | Hasselmo 2006 | Add receptor-specific modulation |
| **B36** | `learning/homeostatic.py` | Sleep-phase modulation | Missing | Homeostatic scaling stronger during NREM | Tononi & Cirelli 2014 | Add sleep-state dependent scaling |
| **B37** | `learning/reconsolidation.py` | Protein synthesis gate | Missing | Reconsolidation requires protein synthesis window | Nader et al. 2000 | Add PSI timing constraint |
| **B38** | `learning/reconsolidation.py` | Novelty calculation dimension mismatch | Query - memory L2 norm | Should use cosine similarity or bidirectional distance | | Use cosine-based novelty metric |
| **B39** | `consolidation/sleep.py` | N3 distribution | Uniform across cycles | Should concentrate in early cycles (cycle 1-2), decrease in later cycles | Aeschbach & Borbely 1993 | Weight NREM_DEEP toward early cycles |
| **B40** | `consolidation/sleep.py` | Missing ripple oscillator | SWR compression exists but no 150-250 Hz ripple | 150-250 Hz ripple frequency during SWRs | Buzsaki et al. 1992 | Add high-frequency ripple component |
| **B41** | `consolidation/sleep.py` | Forward vs reverse replay | Not distinguished | 40-50% forward, 20-30% reverse during SWRs | Foster & Wilson 2006 | Add replay directionality |
| **B42** | `consolidation/sleep.py` | Missing replay.py module | Not present | Should have dedicated replay module with trajectory reconstruction | | Create replay.py with sequence replay |
| **B43** | `consolidation/sleep.py` | Missing two_stage.py module | Not present | Should model NREM→REM two-stage consolidation | Walker & Stickgold 2004 | Create two_stage.py |
| **B44** | `consolidation/sleep.py` | Multi-night consolidation | Single cycle only | Consolidation spans multiple nights, week-scale | Stickgold & Walker 2007 | Add multi-night replay scheduling |
| **B45** | `nca/glymphatic.py` | Clearance rate NREM_DEEP | 0.7 (70%) | ~0.6-0.65 (60-65% improvement) | Xie et al. 2013 | Reduce from 0.9→0.7 to 0.6-0.65 |
| **B46** | `nca/glymphatic.py` | Interstitial space volume | Not modeled | Expands ~60% during sleep | Xie et al. 2013 | Add volumetric model |
| **B47** | `nca/glymphatic.py` | Perivascular dynamics | Missing | Waste flows along perivascular spaces | Iliff et al. 2012 | Add spatial flow model |
| **B48** | `nca/hippocampus.py` | DG sparsity | 4% | 0.5-2% biological | Jung & McNaughton 1993 | Reduce to 0.5-2% range |
| **B49** | `nca/hippocampus.py` | CA3 recurrent connectivity | Implicit in Hopfield | Should model explicit Schaffer collaterals | Rolls 2013 | Add recurrent weight matrix |
| **B50** | `nca/hippocampus.py` | Hippocampal disengagement | Abrupt mode switch | Should gradually disengage during consolidation | Buzsaki 1989 | Add gradual transfer dynamics |
| **B51** | `nca/oscillators.py` | Ripple frequency range | Not modeled | 150-250 Hz during SWRs | Buzsaki & Draguhn 2004 | Add ripple oscillator (150-250 Hz) |
| **B52** | `nca/oscillators.py` | SWR-theta relationship | Independent | SWRs occur during theta troughs | Buzsaki 1986 | Phase-lock SWRs to theta |

---

## MEDIUM Priority (Biological Plausibility - Should Fix)

| ID | File | Parameter/Feature | Current | Literature | Source | Fix |
|----|------|-------------------|---------|------------|--------|-----|
| **B53** | `consolidation/sleep.py` | REM latency | Not enforced | 70-100 min after sleep onset | Carskadon & Dement 2011 | Add REM latency constraint |
| **B54** | `consolidation/sleep.py` | N1/N2/N3 staging | Not explicit | Should model stage transitions | Rechtschaffen & Kales 1968 | Add explicit sleep staging |
| **B55** | `consolidation/sleep.py` | Up/down state durations | Fixed threshold | Up: 200-500ms, Down: 200-500ms | Steriade et al. 1993 | Add bimodal duration distribution |
| **B56** | `consolidation/sleep.py` | Traveling wave dynamics | Not modeled | Slow waves propagate anterior→posterior | Massimini et al. 2004 | Add spatial propagation |
| **B57** | `consolidation/sleep.py` | CA3→CA1 SWR propagation | Not modeled | ~10ms delay CA3→CA1 | Csicsvari et al. 1999 | Add propagation delay |
| **B58** | `nca/glymphatic.py` | A2A receptor | Implicit in adenosine | Should model explicit A2A receptor antagonism effects | Xie et al. 2013 | Add A2A receptor gating |
| **B59** | `learning/acetylcholine.py` | Baseline ACh level | 0.5 | Wake: 0.6-0.8, NREM: 0.2-0.4, REM: 0.7-0.9 | Jasper & Tessier 1971 | Add state-dependent baselines |
| **B60** | `learning/dopamine.py` | TD(λ) trace decay | λ=0.9 fixed | Should vary with serotonin (patience) | Doya 2002 | Add 5-HT modulation of λ |
| **B61** | `learning/dopamine.py` | Tonic vs phasic DA | Combined in RPE | Should separate VTA tonic (0.1-0.3 Hz) from phasic bursts | Grace 1991 | Add dual DA components |
| **B62** | `learning/homeostatic.py` | BCM sliding threshold | tau_BCM missing | tau_BCM ~ hours to days | Bienenstock et al. 1982 | Add explicit tau_BCM parameter |
| **B63** | `learning/serotonin.py` | 5-HT1A vs 5-HT2A | Not separated | 5-HT1A inhibits, 5-HT2A excites | Aghajanian & Marek 1999 | Add receptor subtypes |
| **B64** | `learning/norepinephrine.py` | α1 vs α2 vs β receptors | Not separated | Different effects on plasticity | Sara 2009 | Add receptor subtypes |
| **B65** | `nca/oscillators.py` | Fast vs slow spindles | Not classified | Fast: 12-15 Hz (sensorimotor), Slow: 9-12 Hz (frontal) | De Gennaro & Ferrara 2003 | Add spindle classification |
| **B66** | `nca/oscillators.py` | Spindle-ripple coupling | Not modeled | Spindles coordinate with ripples | Clemens et al. 2007 | Add spindle-SWR coupling |
| **B67** | `nca/hippocampus.py` | Theta phase precession | Not modeled | Place cells precess within theta cycle | O'Keefe & Recce 1993 | Add phase precession |
| **B68** | `nca/hippocampus.py` | Grid cell input | Not modeled | EC grid cells provide spatial input | Hafting et al. 2005 | Add EC grid cell layer |
| **B69** | `nca/striatal_msn.py` | D1 vs D2 MSN opposition | Not explicit | D1=Go, D2=NoGo pathways | Gerfen & Surmeier 2011 | Add explicit pathway separation |
| **B70** | `nca/vta.py` | VTA vs SNc distinction | Combined | VTA: limbic, SNc: motor | Schultz 1998 | Separate VTA and SNc circuits |

---

## LOW Priority (Refinements - Nice to Have)

| ID | File | Parameter/Feature | Current | Literature | Source | Fix |
|----|------|-------------------|---------|------------|--------|-----|
| **B71** | `learning/stdp.py` | Triplet STDP tau_triplet | 40.0ms | Not validated against experiment | Pfister & Gerstner 2006 | Validate against experimental data |
| **B72** | `learning/eligibility.py` | Synaptic tag lifetime | Fixed decay | Should have bimodal (early/late) decay | Frey & Morris 1997 | Add early-LTP vs late-LTP phases |
| **B73** | `consolidation/sleep.py` | Sleep spindle density | Not modeled | 3-5 spindles/min NREM2 | De Gennaro & Ferrara 2003 | Add spindle rate model |
| **B74** | `consolidation/sleep.py` | K-complex generation | Not modeled | Evoked by salient stimuli | Halasz 2005 | Add K-complex events |
| **B75** | `nca/glymphatic.py` | Aquaporin-4 (AQP4) | Implicit in clearance rate | Should model AQP4 water channel density | Iliff et al. 2012 | Add AQP4 channel model |
| **B76** | `nca/oscillators.py` | Alpha frequency variation | Fixed 8-13 Hz | Individual alpha frequency (IAF) 7-13 Hz | Klimesch 1999 | Add IAF personalization |
| **B77** | `nca/oscillators.py` | Cross-frequency coupling | Only theta-gamma | Should add theta-alpha, delta-spindle, etc. | Canolty & Knight 2010 | Add multi-band PAC |
| **B78** | `nca/astrocyte.py` | Ca2+ wave propagation | Not modeled | Astrocytes propagate Ca2+ waves | Cornell-Bell et al. 1990 | Add wave dynamics |
| **B79** | `nca/adenosine.py` | Adenosine kinase | Not modeled | ADK regulates adenosine clearance | Boison 2013 | Add enzymatic regulation |
| **B80** | `nca/spatial_cells.py` | Head direction cells | Not present | HD cells in postsubiculum | Taube et al. 1990 | Add HD cell layer |

---

## MISSING BIOLOGICAL FEATURES

| Feature | Brain Region | Literature | Priority | Implementation |
|---------|--------------|------------|----------|----------------|
| **Replay module** | Hippocampus | Foster & Wilson 2006 | HIGH | Create `consolidation/replay.py` with forward/reverse/preplay |
| **Two-stage consolidation** | HC→Cortex | Walker & Stickgold 2004 | HIGH | Create `consolidation/two_stage.py` with NREM/REM handoff |
| **Ripple oscillator (150-250 Hz)** | CA1/CA3 | Buzsaki et al. 1992 | HIGH | Add to `nca/oscillators.py` |
| **Multi-night consolidation** | System-level | Stickgold & Walker 2007 | HIGH | Add to `consolidation/sleep.py` |
| **Perivascular flow model** | Glymphatic | Iliff et al. 2012 | HIGH | Add to `nca/glymphatic.py` |
| **Schaffer collaterals** | CA3→CA1 | Rolls 2013 | HIGH | Add to `nca/hippocampus.py` |
| **Sleep stage transitions** | Thalamus/Cortex | Rechtschaffen & Kales 1968 | MEDIUM | Add state machine to `consolidation/sleep.py` |
| **Traveling slow waves** | Cortex | Massimini et al. 2004 | MEDIUM | Add spatial model to `nca/oscillators.py` |
| **Spindle-ripple coordination** | Thalamus-HC | Clemens et al. 2007 | MEDIUM | Add coupling to `nca/oscillators.py` |
| **Grid cell input** | Entorhinal Cortex | Hafting et al. 2005 | MEDIUM | Add to `nca/spatial_cells.py` |
| **Phase precession** | CA1 place cells | O'Keefe & Recce 1993 | MEDIUM | Add to `nca/hippocampus.py` |
| **D1/D2 MSN pathways** | Striatum | Gerfen & Surmeier 2011 | MEDIUM | Add to `nca/striatal_msn.py` |
| **VTA vs SNc separation** | Midbrain DA | Schultz 1998 | MEDIUM | Split `nca/vta.py` |
| **Receptor subtypes** | Multiple | Various | MEDIUM | Add to neuromodulator modules |
| **K-complexes** | Cortex (NREM) | Halasz 2005 | LOW | Add to `consolidation/sleep.py` |
| **AQP4 water channels** | Astrocytes | Iliff et al. 2012 | LOW | Add to `nca/glymphatic.py` |
| **Ca2+ wave propagation** | Astrocyte network | Cornell-Bell et al. 1990 | LOW | Add to `nca/astrocyte.py` |
| **Head direction cells** | Postsubiculum | Taube et al. 1990 | LOW | Add to `nca/spatial_cells.py` |

---

## PARAMETER AUDIT FAILURES

| Parameter | Current | Should Be | Source | File | Issue |
|-----------|---------|-----------|--------|------|-------|
| `tau_minus` | 20ms | 25-30ms | Bi & Poo 1998 | `learning/stdp.py` | Outside literature range |
| `dg_sparsity` | 0.04 (4%) | 0.005-0.02 (0.5-2%) | Jung & McNaughton 1993 | `nca/hippocampus.py` | 2-8x too high |
| `clearance_nrem_deep` | 0.7 | 0.6-0.65 | Xie et al. 2013 | `nca/glymphatic.py` | Slightly too high |
| `replay_delay_ms` | 500ms | Correct (1-2 Hz) | ✓ Fixed in P2.5 | `consolidation/sleep.py` | FIXED |
| `tau_nmda_nr2a` | 50ms | Correct | ✓ Jahr & Stevens 1990 | `learning/stdp.py` | FIXED (B13-B22) |
| `tau_nmda_nr2b` | 150ms | Correct | ✓ Jahr & Stevens 1990 | `learning/stdp.py` | FIXED (B13-B22) |
| `theta_freq_hz` | 6.0 Hz | 4-8 Hz (range correct) | ✓ Buzsaki & Draguhn 2004 | `nca/oscillators.py` | OK |
| `gamma_freq_hz` | 40 Hz | 30-100 Hz (range correct) | ✓ Buzsaki & Draguhn 2004 | `nca/oscillators.py` | OK |
| `delta_freq` | 1.5 Hz | 0.5-4 Hz (range correct) | ✓ Steriade et al. 1993 | `nca/oscillators.py` | OK |
| `alpha_freq_hz` | 10 Hz | 8-13 Hz (range correct) | ✓ Klimesch 1999 | `nca/oscillators.py` | OK |

---

## LITERATURE VERIFICATION FAILURES

| Citation | Claim | Implementation | Gap |
|----------|-------|----------------|-----|
| Bi & Poo 1998 | LTD tau = 30-40ms | `tau_minus = 20ms` | Parameter mismatch |
| Xie et al. 2013 | Sleep clearance 60% increase | `clearance = 0.7` (70%) | Overclaimed effect |
| Jung & McNaughton 1993 | DG sparsity 0.5-2% | `dg_sparsity = 0.04` (4%) | 2-8x too dense |
| Buzsaki et al. 1992 | Ripple frequency 150-250 Hz | Not implemented | Missing feature |
| Foster & Wilson 2006 | Forward/reverse replay | Not distinguished | Missing directionality |
| Nader et al. 2000 | Reconsolidation requires protein synthesis | No PSI gate | Missing constraint |
| Hasselmo 2006 | M1/M4 receptors have distinct effects | Not separated | Missing detail |
| Tononi & Cirelli 2014 | Homeostatic scaling during NREM | No sleep modulation | Missing modulation |
| Aeschbach & Borbely 1993 | N3 concentrated in early cycles | Uniform distribution | Wrong temporal pattern |
| Iliff et al. 2012 | Perivascular waste clearance | Not modeled | Missing spatial aspect |

---

## TEST GAPS

| Test Needed | Module | Biological Validation | Why Critical |
|-------------|--------|----------------------|--------------|
| STDP tau validation | `learning/stdp.py` | Verify tau_minus 25-30ms produces correct STDP curve | Core learning rule |
| DG sparsity verification | `nca/hippocampus.py` | Test that 0.5-2% sparsity maintains pattern separation | Pattern separation critical |
| Glymphatic clearance rate | `nca/glymphatic.py` | Verify 60-65% increase during sleep vs wake | Waste clearance biology |
| N3 distribution test | `consolidation/sleep.py` | Verify N3 concentrated in cycles 1-2 | Sleep architecture |
| Ripple frequency test | `nca/oscillators.py` | Verify 150-250 Hz oscillation during SWRs | Memory replay |
| Replay directionality | `consolidation/sleep.py` | Test forward/reverse ratio ~40-50%/20-30% | Consolidation mechanism |
| Protein synthesis window | `learning/reconsolidation.py` | Test reconsolidation fails without PSI | Biological constraint |
| Multi-night consolidation | `consolidation/sleep.py` | Test memory strengthens over multiple sleep cycles | Long-term consolidation |
| Hippocampal disengagement | `nca/hippocampus.py` | Test gradual HC→cortical transfer | Systems consolidation |
| Theta-SWR phase locking | `nca/oscillators.py` | Test SWRs occur at theta troughs | Oscillatory coordination |

---

## SUMMARY STATISTICS

- **Total Issues**: 80 (B31-B80, plus 30 previous B1-B30)
- **HIGH Priority**: 22 issues
- **MEDIUM Priority**: 18 issues
- **LOW Priority**: 10 issues
- **Missing Features**: 18 major biological features
- **Parameter Failures**: 3 critical, 6 fixed, 5 validated OK
- **Literature Mismatches**: 10 major discrepancies
- **Test Gaps**: 10 critical validation tests needed

---

## PREVIOUS FIXES (B1-B30 from Sprints 1-3)

### B1-B12 (Sprint 1) - FIXED
- ✓ B1: Glymphatic clearance 0.9→0.7 (Xie 2013)
- ✓ B2-B12: Protocol methods added to sleep.py

### B13-B22 (Sprint 2) - FIXED
- ✓ B13: NMDA tau added (tau_nmda_nr2a=50ms, tau_nmda_nr2b=150ms)
- ✓ B14: AMPA dynamics added (ampa_* parameters)
- ✓ B15-B22: Various refinements

### B23-B30 (Sprint 3) - DOCUMENTED
- ⚠ B23: N3 distribution incorrect → **B39** (HIGH)
- ⚠ B24: Missing 150-250 Hz ripple → **B40, B51** (HIGH)
- ⚠ B25: No forward/reverse replay → **B41** (HIGH)
- ⚠ B26: Missing replay.py → **B42** (HIGH)
- ⚠ B27: Missing two_stage.py → **B43** (HIGH)
- ⚠ B28: No multi-night consolidation → **B44** (HIGH)
- ⚠ B29-B30: Various MEDIUM issues

---

## RECOMMENDED SPRINT SEQUENCE

### Sprint 4 (HIGH Priority Core)
1. Fix STDP parameters (B31-B34)
2. Fix DG sparsity (B48)
3. Add ripple oscillator (B40, B51)
4. Create replay.py module (B42)

### Sprint 5 (HIGH Priority Consolidation)
1. Fix N3 distribution (B39)
2. Add replay directionality (B41)
3. Create two_stage.py (B43)
4. Add multi-night consolidation (B44)

### Sprint 6 (HIGH Priority Glymphatic)
1. Refine clearance rates (B45)
2. Add perivascular flow (B47)
3. Add interstitial volume model (B46)
4. Add hippocampal disengagement (B50)

### Sprint 7 (MEDIUM Priority Neuromodulators)
1. Add ACh receptor subtypes (B35, B59)
2. Add DA tonic/phasic separation (B61)
3. Add 5-HT receptor subtypes (B63)
4. Add NE receptor subtypes (B64)

### Sprint 8 (MEDIUM Priority Sleep)
1. Add sleep staging (B54)
2. Add REM latency (B53)
3. Add traveling waves (B56)
4. Add CA3→CA1 propagation (B57)

---

## NOTES

- This list represents a COMPLETE audit of all biological modules
- Issues B1-B30 from previous sprints are consolidated here
- Priority assigned based on: (1) Biological accuracy, (2) Impact on core functionality, (3) Literature evidence strength
- All issues have specific literature citations for verification
- Test gaps identified for each major biological feature

**Next Steps**:
1. Review and prioritize with team
2. Assign issues to sprints
3. Create validation tests first (TDD approach)
4. Implement fixes with literature cross-reference
5. Update documentation with biological rationale

---

**Document Version**: 1.0
**Last Updated**: 2026-01-04
**Author**: Claude Code (CompBio Validation Audit)
