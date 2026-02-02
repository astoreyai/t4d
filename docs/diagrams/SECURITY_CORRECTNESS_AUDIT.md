# T4DM Security and Correctness Audit

**Date**: 2026-01-30
**Scope**: Full codebase audit across 6 subsystems
**Auditors**: Computational neuroscience and security analysis (Claude Opus 4.5)

---

## 1. Executive Summary

This audit covers the entire T4DM (formerly World Weaver) codebase across six subsystem areas:

1. **Hippocampal circuit** (hippocampus.py, DG/CA3/CA1/subiculum)
2. **Neuromodulator systems** (VTA, LC, NBM, SNc, crosstalk, dopamine integration)
3. **Sleep/consolidation** (sleep.py, service.py, SWR coupling, spindles, adenosine, lability)
4. **Learning subsystems** (Forward-Forward, capsules, STDP, theta-gamma, coupling)
5. **Memory/storage/API** (Qdrant, Neo4j, WAL, checkpoints, saga, REST API)
6. **Striatal/connectome/spatial** (striatal MSN, connectome, spatial cells, energy, stability, neural field, attention, glial)

### Findings by Severity

| Severity | Count |
|----------|-------|
| **CRITICAL** | 9 |
| **HIGH** | 24 |
| **MEDIUM** | 41 |
| **LOW** | 39+ |

### Top 10 Most Critical Findings

| # | Finding | File:Line | Report |
|---|---------|-----------|--------|
| 1 | Pickle deserialization in checkpoint recovery enables RCE | `persistence/checkpoint.py:180` | Memory/API |
| 2 | No provenance/authentication on any hippocampal operation | `nca/hippocampus.py` (all methods) | Hippocampal |
| 3 | ACh/NE levels externally settable without authentication | `nca/hippocampus.py:1096-1125` | Hippocampal |
| 4 | `receive_pfc_modulation` permanently mutates VTA config | `nca/vta.py:672` | Neuromodulators |
| 5 | False reward injection via `process_memory_outcome` | `nca/dopamine_integration.py:266` | Neuromodulators |
| 6 | `trigger_replay()` allows arbitrary pattern injection during SWR | `nca/swr_coupling.py:540` | Sleep |
| 7 | `force_swr()` bypasses all biological gating | `nca/swr_coupling.py:624-629` | Sleep |
| 8 | No integrity checking on memories before/after consolidation | `consolidation/sleep.py`, `service.py`, `fes_consolidator.py` | Sleep |
| 9 | STDP timing manipulation allows controlled LTP/LTD direction | `learning/stdp.py` record_spike(), `consolidation/stdp_integration.py` | Learning |
| 10 | WAL uses non-cryptographic CRC32, trivially forgeable | `persistence/wal.py:137-179` | Memory/API |

---

## 2. CRITICAL Findings

### C1. Pickle Deserialization in Checkpoint Recovery (RCE)

- **File**: `/mnt/projects/t4d/t4dm/src/t4dm/persistence/checkpoint.py:180`
- **Description**: Checkpoints use `pickle.loads()` for deserialization. While SHA-256 checksum verification exists (line 172), the checksum is stored alongside the data with no HMAC or signing key. The `verify` parameter can be set to `False` (line 142), bypassing even the checksum.
- **Attack scenario**: An attacker with filesystem write access crafts a malicious pickle payload, computes its SHA-256, and embeds both in a valid checkpoint file. On recovery, arbitrary code execution occurs.
- **Recommended fix**: Replace `pickle.loads()` with a safe format (JSON, msgpack) or add HMAC-based signing with a secret key separate from the checkpoint file.

### C2. No Provenance or Authentication on Any Hippocampal Operation

- **File**: `/mnt/projects/t4d/t4dm/src/t4dm/nca/hippocampus.py` (entire file)
- **Description**: The hippocampal circuit operates as a plain Python object with all methods publicly accessible. No concept of trusted vs untrusted input. Memory formation, neuromodulator levels, oscillator replacement, and pattern storage are all equally accessible.
- **Attack scenario**: Any authenticated API user can manipulate the memory system arbitrarily -- inject false memories, force encoding/retrieval modes, or replace oscillators.
- **Recommended fix**: Add an access control layer distinguishing internal (trusted) from external (untrusted) callers. Mark sensitive methods as private and expose only validated interfaces.

### C3. ACh/NE Levels Externally Settable Without Authentication

- **File**: `/mnt/projects/t4d/t4dm/src/t4dm/nca/hippocampus.py:1096-1125`
- **Description**: `receive_ach()` and `receive_ne()` are public methods that directly set internal neuromodulator levels. Any code with a reference to the `HippocampalCircuit` object can force encoding or retrieval mode at will.
- **Attack scenario**: (1) Set ACh high to force encoding mode, (2) inject a crafted pattern via `ca3.store()`, (3) set ACh low to force retrieval mode, (4) retrieve the injected pattern as a genuine memory.
- **Recommended fix**: Neuromodulator levels should only be settable by registered, validated neuromodulator sources. Add caller authentication or restrict to internal-only interfaces.

### C4. PFC Modulation Permanently Mutates VTA Config

- **File**: `/mnt/projects/t4d/t4dm/src/t4dm/nca/vta.py:672`
- **Description**: `receive_pfc_modulation` with `context="goal"` mutates `self.config.tonic_da_level`. Repeated calls ratchet tonic_da_level upward toward 0.5, and `reset()` restores state but uses the corrupted config value.
- **Attack scenario**: An adversary calling `receive_pfc_modulation(1.0, "goal")` repeatedly permanently elevates baseline DA. After `reset()`, the corruption persists because config was mutated, not state.
- **Recommended fix**: Store tonic modulation in `VTAState`, not `VTAConfig`. Config should be immutable after construction.

### C5. False Reward Injection via `process_memory_outcome`

- **File**: `/mnt/projects/t4d/t4dm/src/t4dm/nca/dopamine_integration.py:266`
- **Description**: `process_memory_outcome` accepts `actual_outcome` directly with no validation that the outcome is genuine. An adversary can call this with `actual_outcome=1.0` on any memory_id.
- **Attack scenario**: Generates large positive RPEs, permanently biasing the DA system upward and corrupting value estimates for all memories.
- **Recommended fix**: Require authenticated reward signals. At minimum, clamp and rate-limit outcome values. Add provenance tracking for reward sources.

### C6. Public `trigger_replay()` Allows Arbitrary Pattern Injection During SWR

- **File**: `/mnt/projects/t4d/t4dm/src/t4dm/nca/swr_coupling.py:540`
- **Description**: `trigger_replay(pattern, memory_id, memory_type)` is a public method with no authentication, caller validation, or integrity checking. The pattern is directly injected into `hippocampus.ca3.pattern_completion` (line 585-586).
- **Attack scenario**: Any component with a reference to `SWRNeuralFieldCoupling` can inject arbitrary neural patterns treated as legitimate hippocampal replay, creating false memory associations.
- **Recommended fix**: Validate that replay patterns originate from the memory store. Add caller authentication and pattern integrity checks.

### C7. `force_swr()` Bypasses All Biological Gating

- **File**: `/mnt/projects/t4d/t4dm/src/t4dm/nca/swr_coupling.py:624-629`
- **Description**: `force_swr()` bypasses all biological gating conditions (ACh threshold, NE threshold, hippocampal activity, refractory period, wake/sleep state gating). Only checks if currently quiescent.
- **Attack scenario**: Combined with C6, an attacker can force SWR events at will and inject false memories at any time, regardless of sleep state.
- **Recommended fix**: Remove `force_swr()` or restrict to debug/test builds. In production, all SWR events must pass biological gating.

### C8. No Integrity Checking on Memories Before/After Consolidation

- **File**: `/mnt/projects/t4d/t4dm/src/t4dm/consolidation/sleep.py`, `service.py`, `fes_consolidator.py` (multiple methods)
- **Description**: No checksums, hashes, signatures, or integrity verification on memory content before or after consolidation. Memory corruption propagates silently through consolidation.
- **Attack scenario**: A corrupted or tampered memory passes through consolidation and creates corrupted semantic entities with no audit trail for detection.
- **Recommended fix**: Add content hashing before/after consolidation. Verify embeddings are within expected distance bounds of source materials.

### C9. STDP Timing Manipulation for Pathway Strengthening

- **File**: `/mnt/projects/t4d/t4dm/src/t4dm/learning/stdp.py` (`record_spike()`), `consolidation/stdp_integration.py` (`apply_stdp_to_sequence()`)
- **Description**: `record_spike()` accepts arbitrary timestamps. An attacker who controls timing of spike events can ensure pre-before-post ordering for target pathways (forcing LTP) or post-before-pre for pathways to weaken (forcing LTD). The consolidation replay queue accepts unvalidated episode ID lists.
- **Attack scenario**: Attacker fabricates spike events with controlled timing to selectively strengthen or weaken specific memory pathways.
- **Recommended fix**: Validate that timestamps are plausible (not in future, not too far in past). Reject physiologically implausible inter-spike intervals. Add provenance to spike events.

---

## 3. HIGH Findings

### H1. No Input Dimension Validation in DG

- **File**: `src/t4dm/nca/hippocampus.py:196`
- **Description**: `ec_input` is cast to float32 but shape is never checked against `config.ec_dim`. Mismatched input silently produces wrong-dimensional output or crashes.
- **Attack scenario**: Oversized vector causes memory exhaustion during expansion step (1024x4096 matmul on unexpected input).
- **Fix**: Validate input dimensions at every layer boundary.

### H2. Direct CA3 Pattern Injection (False Memory)

- **File**: `src/t4dm/nca/hippocampus.py:341-374`
- **Description**: `CA3Layer.store()` accepts any numpy array with no check that it came from DG processing. Direct false memory injection vector.
- **Fix**: Mark patterns with provenance indicating DG processing origin.

### H3. No Runtime Invariant Checks in Hippocampus

- **File**: `src/t4dm/nca/hippocampus.py` (throughout)
- **Description**: Expected invariants never verified: DG sparsity, CA3 normalization, CA1 novelty bounds, dimension matching, NaN/Inf absence.
- **Fix**: Add assertion layer at each component boundary.

### H4. No Audit Trail for Memory Formation

- **File**: `src/t4dm/nca/hippocampus.py`
- **Description**: CA3 patterns have UUIDs but no provenance -- no record of whether formed through DG processing or injected directly.
- **Fix**: Add provenance metadata distinguishing legitimate from direct-access patterns.

### H5. Silent Zero-Vector Degenerate State Propagation

- **File**: `src/t4dm/nca/hippocampus.py` (throughout)
- **Description**: Zero-vector input produces zero through DG, returns unchanged from empty CA3, yields novelty=1.0 in CA1. Repeated zero inputs fill CA3 with zero patterns, degrading all retrieval.
- **Fix**: Detect and reject zero/near-zero vectors at input.

### H6. No Bounds on CA1 Novelty Score Output

- **File**: `src/t4dm/nca/hippocampus.py:596`
- **Description**: If both projected vectors are zero, dot product is 0.0, giving novelty=1.0. Attacker can force encoding mode via zero vectors.
- **Fix**: Add explicit bounds checking and zero-input detection.

### H7. ACh-DA Gating Direction Inverted

- **File**: `src/t4dm/nca/neuromod_crosstalk.py:150`
- **Description**: Code implements high ACh = high DA gate. Threlfell et al. (2012) shows ACh interneuron pause enables DA release -- low ACh should enable DA. Direction is backwards.
- **Fix**: Invert the ACh-DA gating sigmoid.

### H8. DA Delay Buffer Uses Wall-Clock Time

- **File**: `src/t4dm/nca/dopamine_integration.py:59,141`
- **Description**: Default delay of 1000ms (should be 150-300ms). Uses `datetime.now()` instead of simulation time. Creates timing side-channel and correctness issues.
- **Fix**: Use simulation time; reduce default delay to 200ms.

### H9. State Diagram DA Decay Tau Mismatch

- **File**: `docs/diagrams/33_state_neuromod.mmd` vs `src/t4dm/nca/vta.py:70`
- **Description**: Diagram shows tau=2s (burst) and tau=5s (dip). Code uses tau=0.2s for both. 10-25x discrepancy.
- **Fix**: Reconcile diagram with code values.

### H10. Direct `_value_table` Write Bypasses VTA Encapsulation

- **File**: `src/t4dm/nca/dopamine_integration.py:520`
- **Description**: `sync_value_estimates` directly accesses VTA's private `_value_table` dict, bypassing validation. Allows silent corruption of VTA's value function.
- **Fix**: Add accessor method to VTA for value table writes.

### H11. ACh Encoding Gate Bypass via Salience Injection

- **File**: `src/t4dm/nca/nucleus_basalis.py:169`
- **Description**: No authentication on `process_salience()`. Any caller can inject salience=1.0 to force high ACh, forcing encoding mode.
- **Fix**: Validate salience source; rate-limit phasic bursts.

### H12. "Orchestra Coordinator" in Diagram Does Not Exist in Code

- **File**: `docs/diagrams/13_neuromodulation_subsystem.mmd`
- **Description**: Diagram shows unified state blending and homeostatic baseline mechanism. No such component exists.
- **Fix**: Implement the coordinator or remove from diagram.

### H13. Reconsolidation Lability Window Exploitable

- **File**: `src/t4dm/consolidation/lability.py`, `consolidation/sleep.py:1751`
- **Description**: Never-retrieved memories can be reconsolidated during sleep (bypasses biological requirement that retrieval precedes lability). No rate limiting on `on_retrieval`. Lability state is in-memory only, lost on restart.
- **Fix**: Require prior retrieval before allowing reconsolidation. Add rate limiting. Persist lability state.

### H14. Consolidation Trigger Externally Forceable

- **File**: `src/t4dm/consolidation/service.py:555`
- **Description**: `consolidate()` accepts `consolidation_type` with no authorization. `set_adenosine()` allows injecting a fake adenosine system that always returns `should_sleep()=True`.
- **Fix**: Add authorization checks; validate adenosine source.

### H15. Sleep Cycle Phase Manipulation

- **File**: `src/t4dm/nca/swr_coupling.py:631-639`
- **Description**: `set_wake_sleep_mode()`, `set_ach_level()`, `set_ne_level()` are public methods directly manipulating gating state.
- **Fix**: Make setters internal-only; validate caller identity.

### H16. Semantic Drift During Abstraction Undetectable

- **File**: `src/t4dm/consolidation/sleep.py:1928`
- **Description**: Cluster centroids become new concept embeddings. The centroid can be semantically meaningless ("average face" problem). Only guard is confidence threshold 0.7, which is insufficient.
- **Fix**: Add semantic coherence validation (e.g., minimum similarity of centroid to all cluster members).

### H17. Post-Consolidation Integrity Verification Missing

- **File**: `src/t4dm/consolidation/` (all files)
- **Description**: No mechanism to verify memories were not corrupted during consolidation. No checksums, embedding distance bounds, or content hashing.
- **Fix**: Add pre/post consolidation integrity verification.

### H18. Request Size Limit Bypass

- **File**: `src/t4dm/api/server.py:213-233`
- **Description**: `RequestSizeLimitMiddleware` only checks `content-length` header. Absent header (chunked transfer) bypasses check entirely.
- **Fix**: Also enforce limits by counting bytes during streaming reads.

### H19. WAL Uses Non-Cryptographic CRC32

- **File**: `src/t4dm/persistence/wal.py:137-179`
- **Description**: CRC32 is trivially vulnerable to collision attacks. Attacker with filesystem access can modify WAL entries while maintaining valid CRC32.
- **Fix**: Use HMAC-SHA256 with a secret key.

### H20. No Data-at-Rest Encryption

- **File**: `src/t4dm/persistence/` (WAL, checkpoints, fallback cache)
- **Description**: Sensitive memory content stored in plaintext on disk.
- **Fix**: Add AES-256 encryption for persistent data.

### H21. No Mandatory Audit Trail for Memory Mutations

- **File**: `src/t4dm/hooks/memory.py:298-323`
- **Description**: `AuditTrailHook` is optional, example-only, in-memory. Not enabled by default. PUT `/episodes/{id}` can modify content with no record.
- **Fix**: Make audit trail mandatory and persistent.

### H22. Client-Controlled Timestamps

- **File**: `src/t4dm/api/routes/episodes.py:36`
- **Description**: Clients supply arbitrary timestamps. No server-side `created_at` field records true ingestion time.
- **Fix**: Add immutable server-side `ingested_at` timestamp.

### H23. Eligibility Trace Poisoning

- **File**: `src/t4dm/nca/coupling.py:194`
- **Description**: `accumulate_eligibility()` accumulates outer product of NT activations. Attacker controlling NT state loads desired correlations, then triggers reward signal to cement them. Partial reset (0.5 decay) means traces persist.
- **Fix**: Cap trace magnitude; add anomaly detection on trace statistics.

### H24. No Authentication on Learning Update Methods

- **File**: `src/t4dm/nca/coupling.py`, `learning/stdp.py`
- **Description**: Any caller can trigger weight changes with arbitrary parameters. No access control, audit logging, or rate limiting.
- **Fix**: Add access control layer for all weight-modifying methods.

### Additional HIGH from Striatal/Connectome Report

### H25. Dale's Law Validation Warns But Does Not Enforce

- **File**: `src/t4dm/nca/connectome.py`
- **Description**: `validate_dales_law()` logs warnings but allows mixed-sign weights violating Dale's law.
- **Fix**: Optionally raise on violation; enforce sign consistency per region.

### H26. Unbounded Persistent Chains in Energy Landscape

- **File**: `src/t4dm/nca/energy.py`
- **Description**: `_persistent_chains` grows without limit during contrastive divergence. Long training causes memory exhaustion.
- **Fix**: Add max chain count or LRU eviction.

### H27. Neural Field PDE Solver Has No NaN/Inf Detection

- **File**: `src/t4dm/nca/neural_field.py`
- **Description**: If diffusion/reaction terms produce NaN, it propagates silently through the entire field before clamping catches symptoms.
- **Fix**: Add explicit `np.isfinite()` checks after each integration step.

### H28. No Input Sanitization on External Memory IDs

- **File**: `src/t4dm/nca/glymphatic_consolidation_bridge.py`
- **Description**: `tag_for_clearance`, `protect_memory` accept arbitrary string memory IDs. `tag_weak_memories` and `tag_stale_memories` accept arbitrary dicts without schema validation.
- **Fix**: Validate memory ID format; add schema validation for input dicts.

### H29. FF Input Poisoning

- **File**: `src/t4dm/nca/forward_forward.py`
- **Description**: `train_positive()` and `train_negative()` accept any numpy array. No provenance, checksum, or sanitization. Attacker can teach network to classify malicious patterns as real.
- **Fix**: Add input validation, provenance tracking, and anomaly detection.

### H30. Capsule Routing Manipulation

- **File**: `src/t4dm/nca/capsules.py`
- **Description**: Crafted inputs dominate routing since routing coefficients are purely data-driven. No defense against adversarial routing convergence.
- **Fix**: Add routing agreement anomaly detection; cap routing coefficient magnitudes.

### H31. Transform Matrix Poisoning via Pose Learning

- **File**: `src/t4dm/nca/capsules.py:602-698`
- **Description**: `learn_pose_from_routing()` updates `_transform_matrices` based on agreement. Repeated crafted inputs shift transforms. No rate limiting or trust verification.
- **Fix**: Rate-limit pose updates; monitor transform matrix drift.

### H32. No Input Sanitization on Spike Recording

- **File**: `src/t4dm/learning/stdp.py`
- **Description**: `record_spike()` accepts any string entity_id. No validation that entity exists, caller is authorized, or timestamp is plausible. Fabricated spikes are indistinguishable from legitimate ones.
- **Fix**: Validate entity existence and timestamp plausibility.

### H33. Eligibility Trace Accumulation Unbounded

- **File**: `src/t4dm/nca/coupling.py`
- **Description**: `accumulate_eligibility()` adds to trace without maximum cap. High-frequency calls build very large traces; combined with large RPE, amplifies weight updates far beyond normal.
- **Fix**: Add maximum cap on trace magnitude.

---

## 4. Security Threat Model

### 4.1 Memory Injection (False Memory Creation)

| ID | Attack Vector | Severity | File |
|----|--------------|----------|------|
| C2 | Direct hippocampal method access (no auth) | CRITICAL | hippocampus.py |
| C3 | Set ACh high -> inject CA3 pattern -> set ACh low -> retrieve | CRITICAL | hippocampus.py:1096 |
| C6 | Inject patterns during SWR replay | CRITICAL | swr_coupling.py:540 |
| H2 | Direct CA3.store() bypasses DG processing | HIGH | hippocampus.py:341 |
| H29 | Poison FF training with adversarial samples | HIGH | forward_forward.py |
| M-1.1 | Store malformed vectors in Qdrant | MEDIUM | qdrant_store.py:301 |
| M-1.3 | Inject arbitrary metadata to fake consolidation status | LOW | qdrant_store.py:316 |

### 4.2 Memory Falsification (Modifying Existing Memories)

| ID | Attack Vector | Severity | File |
|----|--------------|----------|------|
| C8 | Tamper during consolidation (no integrity checks) | CRITICAL | consolidation/*.py |
| H21 | Modify via PUT /episodes/{id} with no audit trail | HIGH | hooks/memory.py:298 |
| H22 | Backdate memories via client-controlled timestamps | HIGH | api/routes/episodes.py:36 |
| H13 | Exploit lability window for reconsolidation | HIGH | lability.py, sleep.py:1751 |
| M-4.3 | Change consolidated vs raw status via payload update | MEDIUM | qdrant_store.py |
| M-4.4 | Forge provenance links in Neo4j | MEDIUM | consolidation/service.py:888 |

### 4.3 Neuromodulator Manipulation (Forcing Brain States)

| ID | Attack Vector | Severity | File |
|----|--------------|----------|------|
| C4 | Permanently ratchet VTA tonic DA via PFC modulation | CRITICAL | vta.py:672 |
| C5 | Inject false rewards to bias DA system | CRITICAL | dopamine_integration.py:266 |
| H7 | ACh-DA gating inverted (biological error enables exploit) | HIGH | neuromod_crosstalk.py:150 |
| H11 | Bypass ACh encoding gate via salience injection | HIGH | nucleus_basalis.py:169 |
| H10 | Corrupt VTA value table directly | HIGH | dopamine_integration.py:520 |
| M-arousal | Push NE arousal to 1.0 via repeated amygdala threat | MEDIUM | locus_coeruleus.py:875 |
| M-crosstalk | Disable all crosstalk regulation | MEDIUM | neuromod_crosstalk.py:46 |

### 4.4 Learning Poisoning (Corrupting Weight Updates)

| ID | Attack Vector | Severity | File |
|----|--------------|----------|------|
| C9 | Fabricate spike timing to control LTP/LTD | CRITICAL | stdp.py, stdp_integration.py |
| H23 | Load desired correlations into eligibility trace | HIGH | coupling.py:194 |
| H24 | Call weight update methods with arbitrary parameters | HIGH | coupling.py, stdp.py |
| H30 | Craft inputs to steer capsule routing consensus | HIGH | capsules.py |
| H31 | Shift transform matrices via repeated pose learning | HIGH | capsules.py:602 |
| M-ff-lr | Manipulate NT state to triple FF learning rate | MEDIUM | forward_forward_nca_coupling.py |
| M-theta | Force encoding phase permanently via oscillator manipulation | MEDIUM | theta_gamma_integration.py |

### 4.5 Persistence Attacks (WAL/Checkpoint Tampering)

| ID | Attack Vector | Severity | File |
|----|--------------|----------|------|
| C1 | Craft malicious pickle payload in checkpoint -> RCE | CRITICAL | checkpoint.py:180 |
| H19 | Modify WAL entries with valid CRC32 collisions | HIGH | wal.py:137 |
| H20 | Read plaintext WAL/checkpoint data (no encryption) | HIGH | persistence/*.py |
| M-3.4 | Saga compensation failure leaves inconsistent state | MEDIUM | storage/saga.py:20 |
| M-3.5 | Pickle protocol version incompatibility | MEDIUM | checkpoint.py:120 |

### 4.6 API Attacks (External Surface)

| ID | Attack Vector | Severity | File |
|----|--------------|----------|------|
| H18 | Bypass request size limit via chunked encoding | HIGH | api/server.py:213 |
| M-2.2 | Access /metrics without authentication | MEDIUM | api/server.py:279 |
| M-1.2 | Mark unlimited memories as permanent via mark-important | MEDIUM | api/routes/episodes.py:391 |
| L-5.2 | Exploit feedback loop to promote specific memories | LOW | api/routes/episodes.py:483 |
| L-5.3 | Spoof session IDs via X-Session-ID header | LOW | api/routes/episodes.py |

### 4.7 Denial of Service (Resource Exhaustion)

| ID | Attack Vector | Severity | File |
|----|--------------|----------|------|
| H26 | Unbounded persistent chains in energy landscape | HIGH | energy.py |
| H1 | Oversized vector causes matmul memory exhaustion | HIGH | hippocampus.py:196 |
| M-vta-cb | Register thousands of DA callbacks | MEDIUM | vta.py:538 |
| M-nbm | Trigger unlimited phasic bursts (no refractory) | MEDIUM | nucleus_basalis.py:189 |
| M-glymph | O(n^2) pending_clearance removal | MEDIUM | glymphatic_consolidation_bridge.py |
| M-state | Unbounded state accumulation in glymphatic bridge | MEDIUM | glymphatic_consolidation_bridge.py |

---

## 5. Biological Correctness Summary

### 5.1 Hippocampal Circuit -- Grade: B

| Component | Rating | Notes |
|-----------|--------|-------|
| DG pattern separation | B | Correct 4% sparsity target, 4x expansion. Uses soft-shrinkage instead of biologically faithful k-WTA competitive inhibition. Orthogonalization is heuristic, not bio-grounded. |
| CA3 pattern completion | B+ | Modern Hopfield (Ramsauer 2020) correctly implemented. Not truly autoassociative (no learned recurrent weights). FIFO eviction is simplistic. |
| CA1 novelty detection | B | Cosine-distance mismatch is defensible. Linear blending lacks biological nonlinearity of dendritic compartment interaction. |
| Subiculum | D | Trivial gain+normalize. Missing burst firing, spatial coding, boundary vector cells. |
| EC layers | A- | EC-II/III distinction properly supported. In-place normalization is a bug. |
| ACh gating | A | Correctly suppresses CA3->CA1 per Hasselmo (2002). |
| Theta gating | A | Encoding at peak, retrieval at trough matches Hasselmo model. |

**Key deviations**: Soft-shrinkage (not WTA) in DG; linear interpolation (not nonlinear dendritic) in CA1; trivial subiculum.

### 5.2 Neuromodulator Systems -- Grade: B-

| Component | Rating | Notes |
|-----------|--------|-------|
| VTA DA circuit | B+ | Correct TD error (Schultz 1997), firing rates in range. Missing D2 autoreceptor feedback. |
| Locus Coeruleus NE | A- | Excellent Yerkes-Dodson implementation. Alpha-2 autoreceptor correctly implemented. Surprise model well done. |
| Nucleus Basalis ACh | B- | Encoding/attention/plasticity modulation all return identical value (should be distinct). No mode-switching thresholds. |
| Substantia Nigra | C+ | Correctly separated from VTA. `connect_to_striatum` is a stub. |
| Neuromodulator crosstalk | C | ACh-DA gating direction is INVERTED vs biology (Threlfell 2012). Min NT level inconsistent with individual circuits. |
| Dopamine integration | C | DA delay 1000ms is 3-10x too long. Uses wall-clock instead of simulation time. Direct VTA internal access. |

**Key deviations**: ACh-DA gating inverted; DA delay 3-10x too long; DA decay tau discrepancy (0.2s code vs 2s/5s diagram); no Orchestra Coordinator despite being diagrammed.

### 5.3 Sleep/Consolidation -- Grade: A-

| Component | Rating | Notes |
|-----------|--------|-------|
| SWR replay | A | Correct frequency (150-250Hz), 10x compression, 90% reverse. Duration slightly long (130ms vs 50-100ms). |
| Spindle-ripple coupling | C | Spindle generation correct. SWR-spindle nesting is documented in diagrams but NOT implemented in code. |
| NREM/REM cycling | A | 90-min ultradian cycle. NREM-dominant early night. 75/25 split. Full N1->N2->N3->REM->N2 cycling. |
| CLS interleaving | A | 60/40 recent/old mixing. Fast->slow transfer implemented. VAE generative replay is advanced. |
| Synaptic homeostasis | B+ | Prune phase with multiplicative downscaling. Per-node (not global) scaling is minor simplification. |
| Reconsolidation | B | Correct lability window (4-8h, Nader 2000). Bug: never-retrieved memories can be reconsolidated. |

**Key deviations**: Spindle-ripple coupling not implemented (only spindle-delta); reconsolidation allows never-retrieved memories.

### 5.4 Learning Subsystems -- Grade: B+

| Component | Rating | Notes |
|-----------|--------|-------|
| Forward-Forward | B+ | Correct goodness function (Hinton 2022). Gradient is approximate (simplified Hebbian). Missing peer normalization. |
| Capsule Networks | B | Correct squashing (Sabour 2017), dynamic routing. EM routing not implemented. O(n^2) scaling. |
| STDP | A- | Correct timing windows (Bi & Poo 1998), multiplicative formulation (van Rossum 2000), three-factor rule. Rate-based abstraction appropriate for memory system. |
| Theta-gamma integration | A- | Correct Lisman-Idiart model, 4-10 item WM capacity, phase-dependent plasticity gating. |
| Learnable coupling | B | Three learning methods (RPE, energy, homeostatic). Gibbs sampling uses non-seedable RNG. |

**Key deviations**: FF uses simplified gradient; capsules lack EM routing; STDP is rate-based (acceptable for this system).

### 5.5 Striatal/Connectome/Spatial -- Grade: B

| Component | Rating | Notes |
|-----------|--------|-------|
| Striatal MSN | B | D1/D2 Hill functions correct. Lateral inhibition symmetric (should be asymmetric). No STN/GPe. |
| Connectome | B+ | 17-region graph with correct major pathways. Dale's law validates but does not enforce. |
| Spatial cells | B | Grid cell sum-of-cosines (Solstad 2006) correct. No boundary effects. Place cell sigma fixed. |
| Neural field | B | Semi-implicit Euler solver. Assumes isotropic uniform grid. No NaN detection. |
| Energy landscape | B+ | Hopfield modern retrieval correct. Contrastive divergence with unbounded chains. |
| Oscillators | B+ | PAC (Tort 2010) standard implementation. Missing cross-frequency coupling beyond PAC. |

**Key deviations**: Symmetric lateral inhibition; no STN/GPe; Mg2+ block oversimplified; fixed Hessian epsilon.

### 5.6 Memory/Storage/API -- Grade: B+ (Engineering)

Not biology-focused, but engineering quality is generally good. Neo4j injection properly mitigated. Security headers well-configured. Saga pattern correctly implemented (with compensation gaps).

---

## 6. Verifiability Assessment

### Invariants That SHOULD Hold

| # | Invariant | Checked? | Location |
|---|-----------|----------|----------|
| 1 | DG output sparsity <= `dg_sparsity` target | NO | hippocampus.py |
| 2 | CA3 output is unit-normalized | NO | hippocampus.py |
| 3 | CA1 novelty_score in [0, 1] | PARTIAL (clamp at 0) | hippocampus.py:596 |
| 4 | All array dimensions match config | NO | hippocampus.py |
| 5 | No NaN/Inf in any intermediate state | NO | All NCA modules |
| 6 | NT concentrations in [0.05, 0.95] | YES (per-circuit clamp) | VTA, LC, NBM, SNc |
| 7 | NT concentrations in [0.0, 1.0] after crosstalk | PARTIAL (min=0.0 inconsistent) | neuromod_crosstalk.py |
| 8 | DA decay tau matches documented values | NO (10-25x mismatch) | vta.py vs 33_state_neuromod.mmd |
| 9 | Positive goodness > negative goodness (FF) | NOT CHECKED | forward_forward.py |
| 10 | Weight bounds [min, max] maintained | YES | stdp.py, forward_forward.py |
| 11 | Multiplicative STDP prevents weight explosion | YES | stdp.py |
| 12 | SWR frequency in [150, 250] Hz | YES (validated) | swr_coupling.py:121-126 |
| 13 | Memory content unchanged after consolidation read | NO | consolidation/*.py |
| 14 | Lability requires prior retrieval | NO (bypassed) | sleep.py:1751 |
| 15 | Replay patterns originate from memory store | NO | swr_coupling.py:540 |
| 16 | WAL entries not tampered | WEAK (CRC32 only) | wal.py:137 |
| 17 | Checkpoint data not tampered | WEAK (SHA-256 no HMAC) | checkpoint.py:172 |
| 18 | Dale's law (sign consistency per region) | WARN ONLY | connectome.py |
| 19 | PDE solver produces finite values | NO | neural_field.py |
| 20 | Working memory capacity 4-10 items | YES | theta_gamma_integration.py:261 |

**Summary**: Of 20 key invariants, only 5 are properly enforced, 3 have partial/weak enforcement, and 12 are not checked at all.

---

## 7. Remaining Diagram-Code Mismatches

Beyond what was fixed in the previous sync commit, the following NEW mismatches were identified:

| # | Diagram | Code | Mismatch |
|---|---------|------|----------|
| 1 | `33_state_neuromod.mmd`: DA burst decay tau=2s, dip decay tau=5s | `vta.py:70`: tau=0.2s for both | **10-25x discrepancy** |
| 2 | `33_state_neuromod.mmd`: NE hyperarousal threshold ">1.5" | `locus_coeruleus.py`: NE clamped to [0, 1] | **Unreachable threshold** |
| 3 | `33_state_neuromod.mmd`: ACh encoding >0.6, retrieval <0.4 | `nucleus_basalis.py`: Returns raw level, no thresholds | **Thresholds not enforced in code** |
| 4 | `13_neuromodulation_subsystem.mmd`: Orchestra Coordinator | No file | **Component does not exist** |
| 5 | `vta_circuit.mermaid`: NAc GABA feedback to GABA neurons | `vta.py`: No GABA interneuron model | **Feature not implemented** |
| 6 | `sleep_cycle/swr_replay diagrams`: SWR nested in spindle troughs | `swr_coupling.py`, `sleep_spindles.py` | **Spindle-SWR nesting not implemented** (only spindle-delta coupling exists) |
| 7 | Hippocampal diagram: NE encoding gain | `hippocampus.py:1150-1159`: `_apply_ne_encoding_gain()` exists but never called | **Dead code** |

---

## 8. MEDIUM and LOW Findings (Summary Table)

### MEDIUM Findings

| # | Finding | File |
|---|---------|------|
| 1 | DG soft-shrinkage not WTA competitive inhibition | hippocampus.py:236 |
| 2 | DG orthogonalization is heuristic, not bio-grounded | hippocampus.py:250 |
| 3 | CA3 not truly autoassociative (no recurrent weights) | hippocampus.py:399 |
| 4 | Beta parameter override enables retrieval manipulation | hippocampus.py:379 |
| 5 | CA3 energy computation may overflow float32 | hippocampus.py:429 |
| 6 | CA1 familiarity blending lacks biological basis | hippocampus.py:600 |
| 7 | Empty CA3 returns query unchanged (false familiar) | hippocampus.py:392 |
| 8 | Subiculum is trivial (gain + normalize) | hippocampus.py:485 |
| 9 | EC in-place normalization mutates caller arrays | hippocampus.py:723 |
| 10 | NE encoding gain dead code | hippocampus.py:1150 |
| 11 | Oscillator externally injectable | hippocampus.py:951 |
| 12 | Recent patterns list unbounded in practice | hippocampus.py:230 |
| 13 | Degenerate zero-vector state propagation | hippocampus.py |
| 14 | Unbounded VTA DA callback injection (DoS) | vta.py:538 |
| 15 | No DA homeostatic mechanism or budget conservation | vta.py |
| 16 | VTA lacks D2 autoreceptor negative feedback | vta.py |
| 17 | NE arousal can be permanently pushed to 1.0 | locus_coeruleus.py:875 |
| 18 | NE stress input has no ceiling on effect | locus_coeruleus.py |
| 19 | NBM encoding/attention/plasticity return identical value | nucleus_basalis.py:344 |
| 20 | NBM no refractory period for phasic bursts | nucleus_basalis.py:189 |
| 21 | NBM no homeostatic mechanism | nucleus_basalis.py |
| 22 | SNc `connect_to_striatum` is a stub | substantia_nigra.py:344 |
| 23 | Crosstalk min_nt_level=0.0 inconsistent with 0.05 | neuromod_crosstalk.py:47 |
| 24 | `enable_crosstalk` flag disables all regulation | neuromod_crosstalk.py:46 |
| 25 | NE hyperarousal ">1.5" unreachable | 33_state_neuromod.mmd |
| 26 | DA delay buffer uses wall-clock + callback injection | dopamine_integration.py |
| 27 | Spindle-ripple coupling documented but not implemented | swr_coupling.py + sleep_spindles.py |
| 28 | No cryptographic provenance binding | consolidation/service.py:883 |
| 29 | Audit trail in-memory only, lost on restart | consolidation/sleep.py |
| 30 | Consolidation non-deterministic/non-reproducible | consolidation/*.py, swr_coupling.py |
| 31 | Regex entity extraction produces false positives | fes_consolidator.py:28 |
| 32 | FF gradient formula is approximate | forward_forward.py |
| 33 | Adversarial negative generation exposes gradient ascent | forward_forward.py |
| 34 | NT state manipulation steers FF/capsule learning | ff_nca_coupling.py, capsule_nca_coupling.py |
| 35 | O(n^2) capsule routing computation | capsules.py |
| 36 | Global np.random usage (non-reproducible) | neural_ode_capsules.py, coupling.py |
| 37 | Theta-gamma plasticity gating bypass | theta_gamma_integration.py |
| 38 | STDP weights directly accessible via private dict | stdp_integration.py |
| 39 | No embedding dimension/range validation on storage | qdrant_store.py:301 |
| 40 | mark-important allows permanent anti-forgetting | api/routes/episodes.py:391 |
| 41 | /metrics exposed without authentication | api/server.py:279 |
| 42 | Saga compensation failure no automated recovery | storage/saga.py:20 |
| 43 | Pickle protocol version lock-in | checkpoint.py:120 |
| 44 | Consolidated vs raw memories not distinguishable | qdrant_store.py |
| 45 | Provenance links not cryptographically signed | consolidation/service.py |
| 46 | No adversarial embedding detection | retrieval layer |
| 47 | Striatal lateral inhibition symmetric (should be asymmetric) | striatal_msn.py |
| 48 | Connectome weights hardcoded, no citations | connectome.py |
| 49 | Grid cell no boundary effects | spatial_cells.py |
| 50 | Astrocyte Km/Vmax fixed, no excitotoxicity desensitization | astrocyte.py |
| 51 | Glymphatic pending_clearance O(n^2) | glymphatic_consolidation_bridge.py |
| 52 | datetime.fromisoformat no error handling | glymphatic_consolidation_bridge.py |
| 53 | Energy landscape unbounded state accumulation | energy.py |
| 54 | Langevin dynamics no adaptive stepping | energy.py |
| 55 | FF goodness threshold fixed (should be layer-adaptive) | energy.py |
| 56 | Hessian finite differences fixed epsilon | stability.py |
| 57 | Neural field operator splitting order matters | neural_field.py |
| 58 | CFL only checks diffusion, not reaction | neural_field.py |
| 59 | PAC phase binning hardcoded | oscillators.py |
| 60 | WM capacity fixed (no dynamic adjustment) | wm_gating.py |
| 61 | Unified attention alpha fixed | unified_attention.py |
| 62 | Glutamate Mg2+ block oversimplified | glutamate_signaling.py |
| 63 | Hill function K_d=0 division by zero | striatal_msn.py |
| 64 | Connectome NBM placement diagram note | connectome_regions.mermaid |

### LOW Findings (39 items, abbreviated)

Key categories: hardcoded parameters (TAN pause, place cell sigma, WM decay, activation decay), missing features (EM routing, STN/GPe, cross-frequency coupling, peer normalization, voltage-dependent STDP), minor biological simplifications (rate-based STDP, single-compartment calcium, linear habit formation), reproducibility issues (non-seedable RNG in LC/neural_ode_capsules), minor numerical risks (sigmoid overflow, softmax overflow, log-of-negative), and positive findings (Neo4j injection mitigated, security headers configured, delay buffer correct, InfoNCE correct).

---

## 9. Recommended Fix Priority

### P0: Security Critical (Must Fix Before Any Deployment)

| # | Fix | Finding |
|---|-----|---------|
| 1 | **Replace `pickle.loads()` in checkpoint recovery** with safe format or add HMAC signing | C1 |
| 2 | **Add access control layer** to hippocampal circuit methods (at minimum, mark sensitive methods private) | C2, C3 |
| 3 | **Fix PFC modulation** to modify VTAState, not VTAConfig | C4 |
| 4 | **Validate reward signal provenance** in `process_memory_outcome` | C5 |
| 5 | **Restrict `trigger_replay()`** to validated internal callers with pattern integrity checks | C6 |
| 6 | **Remove or restrict `force_swr()`** to debug/test builds only | C7 |
| 7 | **Add memory content hashing** before/after consolidation with integrity verification | C8 |
| 8 | **Validate spike timestamps** (reject future, implausible intervals) and entity IDs | C9 |
| 9 | **Replace CRC32 in WAL** with HMAC-SHA256 | H19 |
| 10 | **Add data-at-rest encryption** for WAL and checkpoint files | H20 |
| 11 | **Fix request size limit** to also enforce on chunked transfers | H18 |

### P1: Correctness Critical (Wrong Biology or Numerical Instability)

| # | Fix | Finding |
|---|-----|---------|
| 1 | **Invert ACh-DA gating** in neuromod_crosstalk to match Threlfell (2012) | H7 |
| 2 | **Replace `datetime.now()`** with simulation time in DA delay buffer; reduce default to 200ms | H8 |
| 3 | **Reconcile DA decay tau** between diagram (2s/5s) and code (0.2s) | H9 |
| 4 | **Add NaN/Inf detection** in neural field PDE solver | H27 |
| 5 | **Enforce Dale's law** (optionally raise on violation) in connectome | H25 |
| 6 | **Implement spindle-ripple coupling** (documented but missing) | M-27 |
| 7 | **Wire `_apply_ne_encoding_gain()`** in hippocampus.py or remove dead code | M-10 |
| 8 | **Fix DG zero-vector handling** (detect and reject) | H5 |
| 9 | **Require prior retrieval** before reconsolidation | H13 |
| 10 | **Either implement or remove Orchestra Coordinator** from diagrams | H12 |
| 11 | **Fix NE hyperarousal threshold** in diagram (>1.5 is unreachable) | M-25 |
| 12 | **Fix EC in-place normalization** (copy input before modifying) | M-9 |

### P2: Best Practice (Improvements)

| # | Fix | Finding |
|---|-----|---------|
| 1 | Add input dimension validation at every layer boundary | H1 |
| 2 | Add provenance to CA3 patterns (DG-processed vs direct) | H2, H4 |
| 3 | Add runtime invariant assertions (sparsity, normalization, bounds) | H3 |
| 4 | Make audit trail mandatory and persistent | H21 |
| 5 | Add immutable server-side `ingested_at` timestamp | H22 |
| 6 | Add accessor method to VTA for value table writes | H10 |
| 7 | Add refractory period to NBM phasic bursts | M-20 |
| 8 | Unify min_nt_level across all modules to 0.05 | M-23 |
| 9 | Add D2 autoreceptor feedback to VTA | M-16 |
| 10 | Bound persistent chains in energy landscape | H26 |
| 11 | Add rate limiting on all learning update methods | H24, H33 |
| 12 | Replace global `np.random` with `default_rng` everywhere | M-36 |
| 13 | Add semantic coherence validation for abstraction centroids | H16 |
| 14 | Authenticate /metrics endpoint | M-41 |
| 15 | Add schema validation for glymphatic bridge inputs | H28 |
| 16 | Add learning drift/anomaly detection | Multiple |
| 17 | Replace soft-shrinkage with k-WTA in DG | M-1 |
| 18 | Fix pending_clearance O(n^2) removal | M-51 |
| 19 | Make consolidation reproducible (seedable RNG) | M-30 |

### P3: Nice to Have

| # | Fix | Finding |
|---|-----|---------|
| 1 | Implement EM routing for capsules | L |
| 2 | Add peer normalization to FF | L |
| 3 | Implement STN/GPe hyperdirect pathway | L |
| 4 | Add voltage-dependent STDP (Clopath 2010) | L |
| 5 | Make subiculum biologically richer | M-8 |
| 6 | Add cross-frequency coupling beyond PAC | L |
| 7 | Scale place cell sigma with environment size | L |
| 8 | Use Strang splitting in neural field operator | M-57 |
| 9 | Make asymmetric lateral inhibition in striatum | M-47 |
| 10 | Dynamic unified attention alpha | M-61 |
| 11 | Full Mg2+ voltage-dependent block model | M-62 |
| 12 | Add Hill function K_d>0 config validation | M-63 |
| 13 | Adaptive Hessian epsilon | M-56 |

---

*Generated from 6 subsystem audit reports covering ~50 source files and ~20 diagram files.*
