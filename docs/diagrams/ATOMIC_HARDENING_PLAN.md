# Atomic Hardening Plan — Security + Correctness + Verifiability

**Date**: 2026-01-30
**Source**: SECURITY_CORRECTNESS_AUDIT.md (9 CRITICAL, 33 HIGH, 64 MEDIUM)
**Constraint**: Surgical edits. Each atom is independently testable.
**Principle**: Fix root causes, not symptoms. Every atom has a test assertion.

---

## Architecture: Defense Layers

Before atoms, the plan introduces 3 new foundational modules that multiple atoms depend on:

```
src/ww/core/
├── validation.py     ← FOUNDATION-1: Input validation + sanitization
├── provenance.py     ← FOUNDATION-2: Memory provenance + integrity
└── access_control.py ← FOUNDATION-3: Internal caller authentication
```

These are created first. All subsequent atoms import from them.

---

## FOUNDATION-1: `src/ww/core/validation.py`

**Purpose**: Single source of truth for all input validation across NCA, learning, and API layers.

```python
# Contents:
def validate_array(arr, *, expected_dim, name, allow_zero=False, dtype=np.float32):
    """Validate numpy array: shape, dtype, NaN/Inf, zero-vector, norm bounds."""
    # Raises ValidationError with descriptive message

def validate_embedding(vec, *, dim, name):
    """Validate embedding vector: dimension, finite, unit-norm tolerance."""

def validate_timestamp(ts, *, max_future_s=5.0, max_past_s=86400*365):
    """Reject future timestamps, implausibly old timestamps."""

def validate_nt_level(level, *, name, floor=0.05, ceiling=0.95):
    """Clamp and warn on out-of-range neuromodulator levels."""

def validate_memory_id(mid):
    """Validate UUID format for memory IDs."""

def validate_spike_interval(prev_ts, curr_ts, *, min_isi_ms=1.0):
    """Reject physiologically implausible inter-spike intervals."""

class ValidationError(ValueError): ...
```

**Test**: `tests/core/test_validation.py` — each function has positive/negative/edge cases.

---

## FOUNDATION-2: `src/ww/core/provenance.py`

**Purpose**: Cryptographic provenance for memories and learning events.

```python
import hmac, hashlib

class ProvenanceRecord:
    content_hash: str        # SHA-256 of content
    origin: str              # "dg_processed", "api_store", "consolidation", "replay"
    created_at: datetime     # Server-side immutable timestamp
    creator_id: str          # Module that created this (e.g., "hippocampus.ca3")
    signature: str           # HMAC-SHA256(content_hash + origin + created_at, secret_key)

def sign_content(content_bytes: bytes, origin: str, secret_key: bytes) -> ProvenanceRecord: ...
def verify_provenance(record: ProvenanceRecord, content_bytes: bytes, secret_key: bytes) -> bool: ...
def hash_embedding(embedding: np.ndarray) -> str: ...
```

**Test**: `tests/core/test_provenance.py` — sign/verify round-trip, tamper detection, key rotation.

---

## FOUNDATION-3: `src/ww/core/access_control.py`

**Purpose**: Distinguish trusted internal callers from untrusted external ones. Not full RBAC — just a caller-token system.

```python
class CallerToken:
    module: str          # e.g., "hippocampus", "consolidation_service"
    trust_level: str     # "internal" | "external"
    capabilities: set    # {"write_ca3", "set_neuromod", "trigger_swr", ...}

class AccessDenied(PermissionError): ...

def require_capability(token: CallerToken, capability: str): ...
    """Raises AccessDenied if token lacks capability."""

# Pre-built tokens for internal modules
HIPPOCAMPUS_TOKEN = CallerToken("hippocampus", "internal", {"write_ca3", "set_neuromod", ...})
CONSOLIDATION_TOKEN = CallerToken("consolidation", "internal", {"trigger_swr", "write_ca3", ...})
API_TOKEN = CallerToken("api", "external", {"read", "store_episodic"})
```

**Test**: `tests/core/test_access_control.py` — capability checks, denial on missing capability.

---

## Phase 1: P0 Security Critical (14 atoms)

### ATOM-P0-1: Replace pickle with safe deserialization
- **File**: `src/ww/persistence/checkpoint.py`
- **Root cause**: `pickle.loads()` at line 180 enables arbitrary code execution
- **Fix**:
  1. Replace `pickle.dumps/loads` with `msgpack.packb/unpackb` (or `json` for non-binary state)
  2. Sign checkpoints with HMAC-SHA256 using key from environment variable `T4DM_CHECKPOINT_KEY`
  3. Remove `verify=False` parameter — verification is always mandatory
  4. Add `_CHECKPOINT_VERSION = 2` header for migration
- **Test assertion**: `test_checkpoint_tamper_detection` — modify 1 byte of checkpoint, verify load raises `IntegrityError`
- **Test assertion**: `test_checkpoint_no_pickle` — verify `pickle` is not imported in checkpoint.py

### ATOM-P0-2: Immutable VTAConfig — PFC modulation to state only
- **File**: `src/ww/nca/vta.py`
- **Root cause**: Line 672 mutates `self.config.tonic_da_level` — config is shared/persistent
- **Fix**:
  1. Add `tonic_modulation: float = 0.0` field to `VTAState` dataclass
  2. Change line 672: `self.state.tonic_modulation = min(0.2, self.state.tonic_modulation + delta)` (cap at 0.2)
  3. Change tonic computation: `effective_tonic = self.config.tonic_da_level + self.state.tonic_modulation`
  4. `reset()` clears `state.tonic_modulation = 0.0`, config remains pristine
  5. Freeze config: add `@dataclass(frozen=True)` to `VTAConfig`
- **Test assertion**: `test_pfc_modulation_preserves_config` — call `receive_pfc_modulation` 1000x, assert `config.tonic_da_level == 0.3`
- **Test assertion**: `test_reset_clears_modulation` — modulate, reset, verify effective tonic == config default

### ATOM-P0-3: Restrict hippocampal sensitive methods
- **Files**: `src/ww/nca/hippocampus.py`
- **Root cause**: All methods public, no caller distinction
- **Fix**:
  1. `receive_ach(level, token: CallerToken)` — require `set_neuromod` capability
  2. `receive_ne(level, token: CallerToken)` — require `set_neuromod` capability
  3. `ca3.store(pattern, token: CallerToken)` — require `write_ca3` capability
  4. `set_oscillator(osc, token: CallerToken)` — require `set_oscillator` capability
  5. Import and use `require_capability` from FOUNDATION-3
  6. Add `validate_array` from FOUNDATION-1 at DG entry point (line 196)
- **Test assertion**: `test_ca3_store_rejects_external` — API_TOKEN cannot call ca3.store()
- **Test assertion**: `test_neuromod_requires_capability` — external caller cannot set ACh/NE
- **Test assertion**: `test_dg_rejects_wrong_dimension` — mismatched input raises ValidationError

### ATOM-P0-4: Validate reward signals in dopamine integration
- **File**: `src/ww/nca/dopamine_integration.py`
- **Root cause**: `process_memory_outcome` at line 266 accepts arbitrary outcome values
- **Fix**:
  1. Add `token: CallerToken` parameter, require `submit_reward` capability
  2. Clamp `actual_outcome` to [-1.0, 1.0]
  3. Rate-limit: max 10 outcomes per memory_id per minute (use deque + timestamp)
  4. Add `ProvenanceRecord` for each reward event
  5. Fix `sync_value_estimates` (line 520): replace direct `_value_table` access with `vta.update_value(key, value)` method
- **Test assertion**: `test_reward_clamped` — outcome=100.0 is clamped to 1.0
- **Test assertion**: `test_reward_rate_limited` — 11th call within 60s raises RateLimitError
- **Test assertion**: `test_value_table_encapsulated` — no direct `_value_table` access in codebase

### ATOM-P0-5: Restrict SWR replay and force_swr
- **File**: `src/ww/nca/swr_coupling.py`
- **Root cause**: `trigger_replay` (line 540) and `force_swr` (line 624) are unrestricted public methods
- **Fix**:
  1. `trigger_replay(pattern, memory_id, memory_type, token: CallerToken)` — require `trigger_replay` capability
  2. Add `validate_array(pattern)` before injection
  3. Add `verify_provenance(memory_id)` — pattern must correspond to a stored memory
  4. `force_swr()` → `_force_swr_debug()` — prefix with underscore, add `if not self.config.debug_mode: raise`
  5. In production config: `debug_mode=False` by default
- **Test assertion**: `test_trigger_replay_rejects_external` — API_TOKEN cannot trigger replay
- **Test assertion**: `test_force_swr_disabled_in_production` — raises when debug_mode=False
- **Test assertion**: `test_replay_validates_pattern` — NaN pattern rejected

### ATOM-P0-6: Memory consolidation integrity hashing
- **Files**: `src/ww/consolidation/sleep.py`, `service.py`, `fes_consolidator.py`
- **Root cause**: No integrity checking before/after consolidation
- **Fix**:
  1. Before consolidation read: `pre_hash = hash_embedding(memory.embedding)`
  2. After consolidation read (before processing): verify hash matches stored value
  3. After consolidation write: store new `ProvenanceRecord` with `origin="consolidation"`
  4. `_create_abstraction` (sleep.py:1928): add `origin="rem_abstraction"` provenance
  5. `_consolidate_deep` (service.py:735): add `origin="deep_consolidation"` provenance
  6. FES consolidator: add `origin="fes_consolidation"` provenance
- **Test assertion**: `test_consolidation_detects_tampered_memory` — modify embedding mid-consolidation, verify raises IntegrityError
- **Test assertion**: `test_consolidated_memories_have_provenance` — all outputs have ProvenanceRecord

### ATOM-P0-7: Validate STDP spike timestamps
- **Files**: `src/ww/learning/stdp.py`, `src/ww/consolidation/stdp_integration.py`
- **Root cause**: `record_spike()` accepts arbitrary timestamps and entity IDs
- **Fix**:
  1. `record_spike(entity_id, timestamp, token: CallerToken)` — require `record_spike` capability
  2. Add `validate_timestamp(timestamp)` — reject future, reject >24h past
  3. Add `validate_spike_interval(prev_ts, timestamp)` — reject <1ms ISI
  4. Add `validate_memory_id(entity_id)` — must be valid UUID format
  5. `apply_stdp_to_sequence()`: validate episode IDs exist in memory store before processing
- **Test assertion**: `test_spike_rejects_future_timestamp` — raises ValidationError
- **Test assertion**: `test_spike_rejects_implausible_isi` — 0.1ms interval rejected
- **Test assertion**: `test_stdp_sequence_validates_episodes` — nonexistent episode ID rejected

### ATOM-P0-8: WAL HMAC integrity
- **File**: `src/ww/persistence/wal.py`
- **Root cause**: CRC32 at lines 137-179 is non-cryptographic
- **Fix**:
  1. Replace CRC32 computation with HMAC-SHA256 using key from `T4DM_WAL_KEY` env var
  2. Backward compat: detect CRC32 entries (4-byte checksum) vs HMAC (32-byte), migrate on read
  3. Add `_WAL_VERSION = 2` header
- **Test assertion**: `test_wal_tamper_detection` — flip 1 bit in entry, verify replay raises IntegrityError
- **Test assertion**: `test_wal_v1_migration` — CRC32 entries readable, converted to HMAC on compaction

### ATOM-P0-9: Data-at-rest encryption for persistence
- **Files**: `src/ww/persistence/checkpoint.py`, `wal.py`
- **Root cause**: Plaintext storage of memory content
- **Fix**:
  1. Add `cryptography.fernet` (symmetric encryption) dependency
  2. Encrypt checkpoint data before write, decrypt after read (after HMAC verification)
  3. Encrypt WAL entries before write, decrypt after read (after HMAC verification)
  4. Key from `T4DM_ENCRYPTION_KEY` env var (Fernet key)
  5. Fallback: if no key set, log WARNING and store plaintext (development mode)
- **Test assertion**: `test_checkpoint_encrypted_on_disk` — raw file bytes do not contain known plaintext
- **Test assertion**: `test_wal_encrypted_on_disk` — same

### ATOM-P0-10: Fix chunked transfer size limit bypass
- **File**: `src/ww/api/server.py`
- **Root cause**: Lines 213-233 only check `content-length` header
- **Fix**:
  1. If `content-length` absent, wrap request body in a counting stream reader
  2. Abort with 413 if cumulative bytes exceed `max_request_size` during streaming read
  3. Apply to all non-GET/HEAD/OPTIONS methods
- **Test assertion**: `test_chunked_transfer_enforces_limit` — POST with chunked encoding exceeding limit returns 413

### ATOM-P0-11: Add server-side immutable timestamps
- **File**: `src/ww/api/routes/episodes.py`
- **Root cause**: Line 36 — client-supplied timestamps with no server-side record
- **Fix**:
  1. Add `ingested_at: datetime` field — always set server-side to `datetime.utcnow()`
  2. Client `timestamp` preserved as `event_time` (when the event occurred, per client)
  3. `ingested_at` is immutable — cannot be updated via PUT
  4. Store both in Qdrant payload and Neo4j node properties
- **Test assertion**: `test_ingested_at_server_side` — stored memory has `ingested_at` within 1s of now
- **Test assertion**: `test_ingested_at_immutable` — PUT cannot modify `ingested_at`

### ATOM-P0-12: Restrict SWR phase setters
- **File**: `src/ww/nca/swr_coupling.py:631-639`
- **Root cause**: `set_wake_sleep_mode()`, `set_ach_level()`, `set_ne_level()` are public with no auth — enables sleep cycle phase manipulation (H15)
- **Fix**:
  1. Add `token: CallerToken` to all three methods — require `set_sleep_state` capability
  2. Only `CONSOLIDATION_TOKEN` and `ADENOSINE_TOKEN` have this capability
- **Test assertion**: `test_swr_phase_setters_require_token` — API_TOKEN cannot set wake/sleep mode
- **Test assertion**: `test_consolidation_can_set_sleep_state` — CONSOLIDATION_TOKEN succeeds

### ATOM-P0-13: Protect nucleus basalis salience input
- **File**: `src/ww/nca/nucleus_basalis.py:169`
- **Root cause**: `process_salience()` accepts any value from any caller — enables ACh encoding gate bypass (H11)
- **Fix**:
  1. Add `token: CallerToken` parameter — require `submit_salience` capability
  2. Rate-limit phasic bursts: max 10/minute (use deque + timestamp check)
  3. Clamp salience to [0.0, 1.0]
- **Test assertion**: `test_salience_requires_token` — external caller rejected
- **Test assertion**: `test_salience_rate_limited` — 11th burst in 60s rejected

### ATOM-P0-14: Authorize consolidation triggers
- **File**: `src/ww/consolidation/service.py:555`
- **Root cause**: `consolidate()` accepts `consolidation_type` with no auth, `set_adenosine()` allows injectable adenosine (H14)
- **Fix**:
  1. Add `token: CallerToken` to `consolidate()` — require `trigger_consolidation` capability
  2. Validate `consolidation_type` against whitelist: `{"light", "deep", "full"}`
  3. `set_adenosine()`: require `set_adenosine` capability, validate type is `AdenosineSystem`
  4. Add rate limit: max 1 consolidation per 10 minutes
- **Test assertion**: `test_consolidation_requires_token` — API_TOKEN cannot trigger
- **Test assertion**: `test_consolidation_type_validated` — "evil_type" rejected
- **Test assertion**: `test_set_adenosine_validates_type` — non-AdenosineSystem rejected

---

## Phase 2: P1 Correctness Critical (16 atoms)

### ATOM-P1-1: Invert ACh-DA gating in crosstalk
- **File**: `src/ww/nca/neuromod_crosstalk.py:150`
- **Root cause**: High ACh enables DA — biology says high ACh suppresses DA (Threlfell 2012)
- **Fix**: Change sigmoid: `gate = 1.0 / (1 + exp(5 * (ach - 0.5)))` → `gate = 1.0 / (1 + exp(-5 * (ach - 0.5)))` (invert sign in exponent)
- **Test assertion**: `test_high_ach_suppresses_da` — ACh=0.9 → DA gate < 0.3
- **Test assertion**: `test_low_ach_enables_da` — ACh=0.1 → DA gate > 0.7

### ATOM-P1-2: DA delay buffer to simulation time
- **File**: `src/ww/nca/dopamine_integration.py`
- **Root cause**: Uses `datetime.now()` at lines 79, 92; default delay 1000ms is 3-10x too long
- **Fix**:
  1. Replace `datetime.now()` with `sim_time: float` parameter (seconds since epoch or simulation start)
  2. Change default delay from 1000ms to 200ms (matching Schultz 1997 latency)
  3. All callers pass simulation time explicitly
- **Test assertion**: `test_delay_uses_sim_time` — verify no `datetime.now()` calls in module
- **Test assertion**: `test_default_delay_200ms` — DA arrives after 200ms sim time, not before

### ATOM-P1-3: Reconcile DA decay tau
- **Files**: `docs/diagrams/33_state_neuromod.mmd`, `src/ww/nca/vta.py:70`
- **Root cause**: Diagram says 2s/5s, code says 0.2s — 10-25x discrepancy
- **Decision needed**: Is 0.2s (code) or 2s/5s (diagram) correct?
  - Biology: DA transient lasts 200-500ms in nucleus accumbens (Aragona 2008), 1-2s in PFC (Seamans 2001)
  - **Fix (code-first)**: Change diagram to match code (0.2s), add note "PFC projection uses longer tau"
  - OR add separate `pfc_tau=2.0` for PFC-projecting DA (biologically motivated dual decay)
- **Test assertion**: `test_da_decay_tau_matches_diagram` — diagram value == code value

### ATOM-P1-4: NaN/Inf detection in neural field
- **File**: `src/ww/nca/neural_field.py`
- **Root cause**: No finite-value checks after integration step
- **Fix**:
  1. After each `step()` integration, add: `if not np.all(np.isfinite(self.state)): raise NumericalInstabilityError`
  2. Add same check after diffusion, reaction, and coupling substeps
  3. On detection: log state snapshot, raise descriptive error with last-good state
- **Test assertion**: `test_nan_injection_detected` — inject NaN into coupling, verify raises within 1 step

### ATOM-P1-5: Enforce Dale's law in connectome
- **File**: `src/ww/nca/connectome.py`
- **Root cause**: `validate_dales_law()` warns but allows violations
- **Fix**:
  1. Add `strict_dales_law: bool = True` config parameter
  2. When strict: `validate_dales_law()` raises `DalesLawViolation` instead of warning
  3. Auto-fix mode: project mixed-sign weights to be all-positive or all-negative per region's excitatory/inhibitory classification
- **Test assertion**: `test_dales_law_strict_rejects_mixed_sign` — mixed-sign region raises
- **Test assertion**: `test_dales_law_autofix_projects` — auto-fix produces sign-consistent weights

### ATOM-P1-6: Wire or remove NE encoding gain
- **File**: `src/ww/nca/hippocampus.py:1150-1159`
- **Root cause**: `_apply_ne_encoding_gain()` exists but is never called — dead code
- **Fix**: Wire into `process()` at the DG expansion step (after line 710, before DG.process):
  ```python
  if self._ne_level > 0.3:  # NE high enough to modulate
      ec_input = self._apply_ne_encoding_gain(ec_input)
  ```
- **Test assertion**: `test_ne_gain_modulates_dg_input` — high NE changes DG output vs low NE
- **Test assertion**: `test_ne_gain_threshold` — NE=0.2 does not trigger gain

### ATOM-P1-7: Fix DG zero-vector degenerate state
- **File**: `src/ww/nca/hippocampus.py`
- **Root cause**: Zero input propagates silently, fills CA3 with zeros
- **Fix**:
  1. At `process()` entry: `if np.linalg.norm(ec_input) < 1e-8: raise ValidationError("Zero-vector input")`
  2. At DG output: assert sparsity <= target (add invariant check)
  3. At CA3.store: reject zero/near-zero patterns
- **Test assertion**: `test_zero_vector_rejected` — raises ValidationError
- **Test assertion**: `test_dg_sparsity_invariant` — DG output sparsity <= config.dg_sparsity + 0.01 tolerance

### ATOM-P1-8: Require retrieval before reconsolidation
- **File**: `src/ww/consolidation/sleep.py:1751`
- **Root cause**: `can_reconsolidate = True` when `last_accessed is None` — allows reconsolidation of never-retrieved memories
- **Fix**: Change line 1751:
  ```python
  can_reconsolidate = (last_accessed is not None) and lability_manager.is_labile(memory_id)
  ```
- **Test assertion**: `test_never_retrieved_cannot_reconsolidate` — memory with no retrieval skipped in reconsolidation
- **Test assertion**: `test_retrieved_memory_can_reconsolidate` — memory retrieved within lability window is processed

### ATOM-P1-9: Fix EC in-place normalization
- **File**: `src/ww/nca/hippocampus.py:723-726`
- **Root cause**: `arr[:] = arr / norm` mutates caller's array
- **Fix**: `arr = arr / norm` (create new array, don't write back to caller's view)
- **Test assertion**: `test_ec_normalization_no_mutation` — pass array, verify original unchanged after process()

### ATOM-P1-10: Fix diagram — unreachable NE threshold
- **File**: `docs/diagrams/33_state_neuromod.mmd`
- **Root cause**: Shows ">1.5" for hyperarousal but NE clamped to [0, 1]
- **Fix**: Change to `NE > 0.8 → HYPERAROUSED` (matching LC high tonic threshold)
- **Test assertion**: Diagram review (manual)

### ATOM-P1-11: Fix diagram — ACh encoding/retrieval thresholds
- **File**: `docs/diagrams/33_state_neuromod.mmd`
- **Root cause**: Shows encoding >0.6, retrieval <0.4 — not enforced in code
- **Decision**: Either implement thresholds in NBM code or remove from diagram
- **Fix**: Add to `nucleus_basalis.py`:
  ```python
  def get_cognitive_mode(self) -> str:
      if self.state.ach_level > 0.6: return "ENCODING"
      if self.state.ach_level < 0.4: return "RETRIEVAL"
      return "TRANSITIONAL"
  ```
- **Test assertion**: `test_ach_encoding_threshold` — ACh=0.7 → ENCODING
- **Test assertion**: `test_ach_retrieval_threshold` — ACh=0.3 → RETRIEVAL

### ATOM-P1-12: Fix diagram — DA decay tau OR implement dual-tau
- **File**: `docs/diagrams/33_state_neuromod.mmd`
- **Root cause**: 10-25x discrepancy with code
- **Fix**: Update diagram lines to show `tau=0.2s (synaptic)` and add note `PFC tau=2s (volume, planned)`
- **Test assertion**: Diagram review (manual)

### ATOM-P1-13: Persist lability state and rate-limit retrieval
- **File**: `src/ww/consolidation/lability.py`
- **Root cause**: Lability state in-memory only (lost on restart), no rate limit on `on_retrieval` — repeated calls keep memories permanently labile (H13 partial)
- **Fix**:
  1. Persist `LabilityState` to database (add `save_state()` / `load_state()` using storage backend)
  2. Rate-limit `on_retrieval`: max 100 calls per memory_id per hour (deque + timestamp)
  3. On restart: load persisted lability states
- **Test assertion**: `test_lability_persisted_across_restart` — create labile memory, simulate restart, verify state loaded
- **Test assertion**: `test_retrieval_rate_limited` — 101st retrieval in 1 hour rejected

### ATOM-P1-14: Spindle-ripple coupling implementation
- **Files**: `src/ww/nca/swr_coupling.py`, `src/ww/nca/sleep_spindles.py`
- **Root cause**: Spindle-SWR nesting documented in diagrams but not implemented — SWR events not gated to spindle troughs (M-27)
- **Fix**:
  1. Add `spindle_phase_gate` to SWR initiation: SWR can only trigger when spindle is in trough phase (π ± π/4)
  2. `SpindleDeltaCoupler` gains `get_current_phase()` method
  3. `SWRNeuralFieldCoupling._check_swr_initiation()` queries spindle phase before allowing SWR
- **Test assertion**: `test_swr_gated_to_spindle_trough` — SWR blocked when spindle at peak, allowed at trough
- **Test assertion**: `test_swr_spindle_timing` — SWR onset within 50ms of spindle trough center

---

## Phase 3: P2 Best Practice (26 atoms)

### ATOM-P2-1: Input validation at every NCA layer boundary
- **Files**: `hippocampus.py`, `vta.py`, `locus_coeruleus.py`, `nucleus_basalis.py`, all NCA modules
- **Fix**: Add `validate_array()` call at entry of every `process()` and `step()` method
- **Test**: Each module has `test_{module}_rejects_nan_input`

### ATOM-P2-2: Provenance on CA3 patterns
- **File**: `src/ww/nca/hippocampus.py`
- **Fix**: `CA3Layer.store(pattern, provenance: ProvenanceRecord)` — provenance is mandatory
- **Test**: `test_ca3_store_requires_provenance`

### ATOM-P2-3: Runtime invariant assertion layer
- **File**: New `src/ww/nca/invariants.py`
- **Fix**: Decorator `@check_invariants` that validates pre/post conditions per NCA method:
  - DG: output sparsity <= target
  - CA3: output unit-normalized
  - CA1: novelty in [0, 1]
  - All: no NaN/Inf, dimensions match config
- **Test**: `test_invariant_catches_sparsity_violation`

### ATOM-P2-4: Mandatory persistent audit trail
- **Files**: `src/ww/hooks/memory.py`, `src/ww/api/routes/episodes.py`
- **Fix**: Replace optional in-memory `AuditTrailHook` with mandatory persistent `AuditLog` writing to append-only file
- **Test**: `test_memory_mutation_creates_audit_entry`

### ATOM-P2-5: VTA value table accessor method
- **Files**: `src/ww/nca/vta.py`, `src/ww/nca/dopamine_integration.py`
- **Fix**: Add `vta.update_value_estimate(key, value, token)`. Remove direct `_value_table` access.
- **Test**: `test_no_direct_value_table_access` — grep codebase for `_value_table` outside vta.py

### ATOM-P2-6: NBM phasic refractory period
- **File**: `src/ww/nca/nucleus_basalis.py`
- **Fix**: Add `phasic_refractory_ms: float = 100.0` config. Reject phasic bursts within refractory.
- **Test**: `test_nbm_refractory_blocks_rapid_bursts`

### ATOM-P2-7: Unify min_nt_level to 0.05
- **File**: `src/ww/nca/neuromod_crosstalk.py:47`
- **Fix**: Change `min_nt_level: float = 0.0` to `0.05`
- **Test**: `test_crosstalk_min_level_matches_circuits`

### ATOM-P2-8: VTA D2 autoreceptor feedback
- **File**: `src/ww/nca/vta.py`
- **Fix**: Add D2 autoreceptor using same Hill function pattern as LC alpha-2. High DA suppresses firing.
- **Test**: `test_vta_d2_autoreceptor_negative_feedback` — high DA reduces firing rate

### ATOM-P2-9: Bound persistent chains in energy
- **File**: `src/ww/nca/energy.py`
- **Fix**: Add `max_persistent_chains: int = 1000` config. LRU eviction when exceeded.
- **Test**: `test_persistent_chains_bounded` — after 2000 iterations, chain count <= 1000

### ATOM-P2-10: Rate-limit all learning update methods
- **Files**: `src/ww/nca/coupling.py`, `src/ww/learning/stdp.py`, `src/ww/nca/forward_forward.py`, `src/ww/nca/capsules.py`
- **Fix**: Add `max_updates_per_second: int = 100` config. Track update timestamps. Reject excess.
- **Test**: `test_learning_rate_limited` — 101st update within 1s raises RateLimitError

### ATOM-P2-11: Replace global np.random with default_rng
- **Files**: `src/ww/nca/neural_ode_capsules.py`, `src/ww/nca/coupling.py`, `src/ww/nca/swr_coupling.py`, `src/ww/consolidation/sleep.py`
- **Fix**: Replace all `np.random.random()`, `np.random.randn()`, `random.random()` with `self._rng.random()` using `np.random.default_rng(seed)`
- **Test**: `test_deterministic_with_seed` — same seed produces same output

### ATOM-P2-12: Semantic coherence validation for abstractions
- **File**: `src/ww/consolidation/sleep.py:1928`
- **Fix**: After computing centroid, verify minimum cosine similarity of centroid to EVERY cluster member >= 0.5 (not just average)
- **Test**: `test_abstraction_rejects_incoherent_cluster` — cluster with outlier member rejected

### ATOM-P2-13: Authenticate /metrics endpoint
- **File**: `src/ww/api/server.py:279`
- **Fix**: Remove `/metrics` from `EXEMPT_PATHS`. Require API key for metrics access.
- **Test**: `test_metrics_requires_auth` — GET /metrics without key returns 401

### ATOM-P2-14: Schema validation for glymphatic bridge inputs
- **File**: `src/ww/nca/glymphatic_consolidation_bridge.py`
- **Fix**: Add `validate_memory_id()` to `tag_for_clearance`, `protect_memory`. Add schema validation to dict inputs.
- **Test**: `test_glymphatic_rejects_invalid_id`

### ATOM-P2-15: Cap eligibility trace magnitude
- **File**: `src/ww/nca/coupling.py:194`
- **Fix**: After accumulation: `self._eligibility = np.clip(self._eligibility, -max_trace, max_trace)` where `max_trace=10.0`
- **Test**: `test_eligibility_trace_capped` — 1000 accumulations, trace magnitude <= max_trace

### ATOM-P2-16: Learning drift/anomaly detection
- **Files**: `src/ww/nca/forward_forward.py`, `capsules.py`, `coupling.py`
- **Fix**: Track running mean/std of goodness, agreement, coupling eigenvalues. Alert (log WARNING) when > 3 sigma from baseline.
- **Test**: `test_drift_detection_alerts_on_poisoning` — inject adversarial samples, verify warning logged

### ATOM-P2-17: Replace DG soft-shrinkage with k-WTA
- **File**: `src/ww/nca/hippocampus.py:236-248`
- **Fix**: Replace soft-shrinkage with hard top-k selection:
  ```python
  k = int(self.config.dg_sparsity * len(expanded))
  topk_idx = np.argpartition(np.abs(expanded), -k)[-k:]
  sparse = np.zeros_like(expanded)
  sparse[topk_idx] = expanded[topk_idx]
  ```
- **Test**: `test_dg_kwta_exact_sparsity` — output has exactly k nonzero elements

### ATOM-P2-18: Fix pending_clearance O(n^2)
- **File**: `src/ww/nca/glymphatic_consolidation_bridge.py`
- **Fix**: Change `pending_clearance` from `list` to `collections.OrderedDict` for O(1) removal
- **Test**: `test_clearance_performance` — 10000 items, clear 5000, time < 100ms

### ATOM-P2-19: Seedable consolidation for reproducibility
- **Files**: `src/ww/consolidation/sleep.py`, `src/ww/nca/swr_coupling.py`
- **Fix**: Accept `seed: int | None` in constructors. Pass to all `np.random.default_rng(seed)` calls. None = random.
- **Test**: `test_consolidation_deterministic` — same seed, same input → same output

### ATOM-P2-20: FF input provenance and checksums
- **File**: `src/ww/nca/forward_forward.py`
- **Root cause**: `train_positive()` and `train_negative()` accept any array — no provenance (H29)
- **Fix**: Add `provenance: ProvenanceRecord` parameter to both methods. Store content hash. Reject unsigned data.
- **Test**: `test_ff_training_requires_provenance` — unsigned data rejected

### ATOM-P2-21: Capsule routing coefficient caps
- **File**: `src/ww/nca/capsules.py`
- **Root cause**: Crafted inputs dominate routing — no magnitude cap (H30)
- **Fix**: Cap routing logits to [-5, 5] before softmax. Add anomaly detection: warn if any routing coefficient >4.0 pre-cap.
- **Test**: `test_routing_coefficients_capped` — extreme input produces capped logits
- **Test**: `test_routing_anomaly_detected` — adversarial input triggers warning

### ATOM-P2-22: Transform matrix drift monitoring
- **File**: `src/ww/nca/capsules.py:602-698`
- **Root cause**: `learn_pose_from_routing()` can shift transforms without detection (H31)
- **Fix**: Track Frobenius norm of each transform matrix per update. Warn if delta > 3 sigma from running mean.
- **Test**: `test_transform_drift_detected` — rapid large updates trigger warning

### ATOM-P2-23: STDP weight encapsulation
- **File**: `src/ww/learning/stdp.py`, `src/ww/consolidation/stdp_integration.py`
- **Root cause**: `self.stdp._weights` directly accessed/deleted by consolidation module (M-38)
- **Fix**: Add `get_connections()`, `remove_connection(key)`, `get_weight(key)` methods to STDP class. Remove direct `_weights` access from stdp_integration.py.
- **Test**: `test_no_direct_weights_access` — grep codebase for `._weights` outside stdp.py

### ATOM-P2-24: Qdrant embedding dimension validation
- **File**: `src/ww/storage/qdrant_store.py:301-332`
- **Root cause**: `add()` accepts arbitrary vectors with no dimension/range validation (M-39)
- **Fix**: Add `validate_embedding(vec, dim=self.dimension)` before upsert. Reject NaN/Inf/wrong-dim.
- **Test**: `test_qdrant_rejects_wrong_dim` — 512-dim vector to 1024-dim collection rejected
- **Test**: `test_qdrant_rejects_nan` — NaN vector rejected

### ATOM-P2-25: Memory origin type field
- **File**: `src/ww/storage/qdrant_store.py`, `src/ww/api/routes/episodes.py`
- **Root cause**: Consolidated vs raw memories not distinguishable (M-44)
- **Fix**: Add immutable `origin_type` field to payload: `"episodic_raw"`, `"episodic_consolidated"`, `"semantic"`, `"rem_abstraction"`. Set at creation, immutable via update.
- **Test**: `test_origin_type_set_on_create` — new memory has origin_type
- **Test**: `test_origin_type_immutable` — PUT cannot change origin_type

### ATOM-P2-26: Theta-gamma plasticity gating protection
- **File**: `src/ww/nca/theta_gamma_integration.py`
- **Root cause**: Oscillator manipulation forces permanent encoding mode, boosting learning 2x (M-37)
- **Fix**: Add `max_encoding_streak: int = 10` — if encoding phase active for >10 consecutive theta cycles, auto-reset to balanced mode and log warning.
- **Test**: `test_encoding_streak_limited` — after 10 consecutive encoding cycles, mode auto-resets

---

## Phase 4: P3 Hardening — MEDIUM Findings (39 atoms)

### Hippocampal Circuit (7 atoms)

#### ATOM-P3-1: CA3 energy overflow protection
- **File**: `src/ww/nca/hippocampus.py:429-432`
- **Root cause**: `exp(8.0 * similarities)` can overflow float32 with 1000 patterns (M-5)
- **Fix**: Use `np.float64` for energy computation. Add log-sum-exp trick: subtract max before exp.
- **Test**: `test_ca3_energy_no_overflow` — 1000 high-similarity patterns, no Inf

#### ATOM-P3-2: Beta parameter internal only
- **File**: `src/ww/nca/hippocampus.py:379`
- **Root cause**: `complete(beta=)` accepts external beta — extreme values manipulate retrieval (M-4)
- **Fix**: Remove `beta` parameter from public `complete()`. Use `self.config.beta` only. Add `_complete_internal(beta)` for internal use.
- **Test**: `test_beta_not_externally_settable` — `complete()` signature has no beta param

#### ATOM-P3-3: CA3 empty store returns sentinel
- **File**: `src/ww/nca/hippocampus.py:392`
- **Root cause**: Empty CA3 returns query unchanged — downstream CA1 produces false "familiar" signal (M-7)
- **Fix**: When CA3 has no stored patterns, return `(None, 0, 0.0)` instead of `(query, 0, 0.0)`. Caller handles None as "no match".
- **Test**: `test_empty_ca3_returns_none` — no patterns stored, complete() returns None

#### ATOM-P3-4: Recent patterns list bounded
- **File**: `src/ww/nca/hippocampus.py:230-232`
- **Root cause**: `_recent_patterns` list can grow unbounded in practice (M-12)
- **Fix**: Use `collections.deque(maxlen=self._max_recent)` instead of list with manual trim.
- **Test**: `test_recent_patterns_bounded` — after 200 inputs, len <= max_recent

#### ATOM-P3-5: DG orthogonalization biological grounding note
- **File**: `src/ww/nca/hippocampus.py:250-304`
- **Root cause**: Centroid-projection orthogonalization has no biological basis (M-2)
- **Fix**: Add docstring noting this is a computational approximation. Add config `use_orthogonalization: bool = True` to allow disabling. Not changing the algorithm — just documenting the approximation.
- **Test**: `test_orthogonalization_disableable` — config False skips orthogonalization

#### ATOM-P3-6: CA1 novelty score explicit bounds
- **File**: `src/ww/nca/hippocampus.py:596-601`
- **Root cause**: CA1 blending lacks biological nonlinearity (M-6); novelty unbounded on edge cases
- **Fix**: Clamp novelty_score to [0.0, 1.0] explicitly. Add comment noting linear blending is a simplification of dendritic compartment interaction.
- **Test**: `test_ca1_novelty_always_bounded` — fuzz test with random inputs, all novelty in [0, 1]

#### ATOM-P3-7: Subiculum documentation of limitations
- **File**: `src/ww/nca/hippocampus.py:485-511`
- **Root cause**: Subiculum is trivial gain+normalize — missing burst firing, boundary vector cells (M-8)
- **Fix**: Add docstring: "Simplified subiculum. Missing: burst firing (Sharp 2006), boundary vector cells (Lever 2009), spatial output coding. PLANNED for future enhancement." Add `:::partial` to diagram node if not already.
- **Test**: N/A (documentation-only)

### Neuromodulators (8 atoms)

#### ATOM-P3-8: VTA callback registration limits
- **File**: `src/ww/nca/vta.py:538`
- **Root cause**: Unbounded callback list — DoS via thousands of registrations (M-14)
- **Fix**: `max_callbacks: int = 100` config. Reject with warning when exceeded.
- **Test**: `test_vta_callback_limit` — 101st registration rejected

#### ATOM-P3-9: DA homeostatic regulation
- **File**: `src/ww/nca/vta.py`
- **Root cause**: No long-term homeostatic mechanism for DA (M-15)
- **Fix**: Add slow homeostatic term: if running mean DA (over 1000 steps) > `homeostatic_target * 1.2`, reduce `tonic_rate` by 5% per step until within range. Cite Bhatt et al. 1998 (DA receptor desensitization).
- **Test**: `test_da_homeostasis` — sustained high DA causes tonic rate reduction

#### ATOM-P3-10: NE arousal ceiling with habituation
- **File**: `src/ww/nca/locus_coeruleus.py:875`
- **Root cause**: NE arousal permanently pushable to 1.0 via repeated threat (M-17, M-18)
- **Fix**: Add habituation: if arousal_drive > 0.8 for > 60 simulation-seconds, apply `habituation_factor *= 0.99` per step. Reset habituation when arousal_drive drops below 0.5.
- **Test**: `test_ne_arousal_habituates` — sustained threat reduces effective arousal over time

#### ATOM-P3-11: NBM mode differentiation
- **File**: `src/ww/nca/nucleus_basalis.py:344-376`
- **Root cause**: `get_encoding_modulation`, `get_attention_modulation`, `get_plasticity_gate` all return identical value (M-19)
- **Fix**: Differentiate: encoding = `ach_level`, attention = `ach_level * 0.7 + 0.3 * phasic_component`, plasticity = `ach_level * sigmoid(phasic_burst_recency)`. Cite Hasselmo 2006 for distinct ACh functions.
- **Test**: `test_nbm_modes_produce_different_values` — all three return distinct values for same ACh level

#### ATOM-P3-12: NBM homeostatic mechanism
- **File**: `src/ww/nca/nucleus_basalis.py`
- **Root cause**: No homeostatic mechanism — ACh stays high if salience sustained (M-21)
- **Fix**: Add desensitization: sustained ACh > 0.7 for >30s triggers receptor desensitization (reduce effective ACh by 10% per 10s). Recovers when ACh drops below 0.5.
- **Test**: `test_nbm_desensitization` — sustained salience reduces effective ACh over time

#### ATOM-P3-13: SNc connect_to_striatum implementation
- **File**: `src/ww/nca/substantia_nigra.py:331`
- **Root cause**: `connect_to_striatum()` is a stub — `pass` only (M-22)
- **Fix**: Implement D1/D2 modulation: `connect_to_striatum(striatum: StriatalMSN)` stores reference. In `step()`, modulate striatum D1/D2 excitability via `striatum.receive_da(self.state.da_level)`.
- **Test**: `test_snc_modulates_striatum` — high SNc DA increases D1 excitability

#### ATOM-P3-14: Crosstalk enable flag protection
- **File**: `src/ww/nca/neuromod_crosstalk.py:46`
- **Root cause**: `enable_crosstalk` flag disables all cross-system regulation (M-24)
- **Fix**: Make `enable_crosstalk` read-only after construction. Remove setter. Log WARNING if constructed with `False`.
- **Test**: `test_crosstalk_flag_immutable` — cannot change flag after construction

#### ATOM-P3-15: DopamineIntegration callback limits
- **File**: `src/ww/nca/dopamine_integration.py`
- **Root cause**: Unbounded callback registration (same pattern as VTA) (M-26 partial)
- **Fix**: Same as P3-8 — max 100 callbacks for both `rpe_callbacks` and `da_callbacks`.
- **Test**: `test_da_integration_callback_limit` — 101st rejected

### Learning Systems (5 atoms)

#### ATOM-P3-16: Regex entity extraction improvement
- **File**: `src/ww/consolidation/fes_consolidator.py:28-76`
- **Root cause**: Basic regex produces high false-positive entity extractions (M-31)
- **Fix**: Add confidence scoring to extracted entities: file patterns require path separator, function patterns require alphanumeric, name patterns require ≥2 words. Filter entities below `min_confidence=0.5`.
- **Test**: `test_entity_extraction_filters_noise` — random code fragment produces <5 false entities

#### ATOM-P3-17: FF adversarial negative generation restriction
- **File**: `src/ww/nca/forward_forward.py`
- **Root cause**: `generate_negative("adversarial")` exposes gradient ascent — attacker uses to craft backdoors (M-33)
- **Fix**: Make adversarial generation internal-only (`_generate_negative_adversarial`). Public `generate_negative()` only supports `"noise"`, `"shuffle"`, `"wrong_label"`, `"hybrid"`.
- **Test**: `test_adversarial_generation_not_public` — `generate_negative("adversarial")` raises ValueError

#### ATOM-P3-18: Pickle protocol elimination (checkpoint version compat)
- **File**: `src/ww/persistence/checkpoint.py:120`
- **Root cause**: `pickle.HIGHEST_PROTOCOL` creates version-locked checkpoints (M-43)
- **Fix**: Already addressed by ATOM-P0-1 (replace pickle with msgpack). This atom ensures msgpack version field in header for forward compat.
- **Test**: `test_checkpoint_version_field` — checkpoint contains version header

#### ATOM-P3-19: Coupling Gibbs sampling reproducibility
- **File**: `src/ww/nca/coupling.py:432`
- **Root cause**: `_gibbs_sample()` uses global `np.random.randn()` — non-reproducible (M-36 extension)
- **Fix**: Already covered by P2-11 for most files. This specifically targets `coupling.py` Gibbs sampling to use `self._rng`.
- **Test**: `test_gibbs_sampling_deterministic` — same seed, same samples

#### ATOM-P3-20: FF goodness threshold layer-adaptive
- **File**: `src/ww/nca/forward_forward.py`
- **Root cause**: Fixed threshold for positive/negative data — should be per-layer (M-55)
- **Fix**: Add `adaptive_threshold: bool = True` config. When True, theta per layer = running mean of (positive_goodness + negative_goodness) / 2.
- **Test**: `test_ff_adaptive_threshold` — different layers converge to different thresholds

### Storage/API (5 atoms)

#### ATOM-P3-21: mark-important rate limiting
- **File**: `src/ww/api/routes/episodes.py:391-437`
- **Root cause**: Unlimited memories can be marked permanent, defeating forgetting (M-40)
- **Fix**: Add `max_important_memories: int = 1000` system config. Add rate limit: max 10 mark-important per hour.
- **Test**: `test_mark_important_limited` — 1001st mark rejected
- **Test**: `test_mark_important_rate_limited` — 11th in 1 hour rejected

#### ATOM-P3-22: Saga auto-recovery for compensation failures
- **File**: `src/ww/storage/saga.py:20-52`
- **Root cause**: `CompensationError` only logged, requires manual fix (M-42)
- **Fix**: Add dead-letter queue: failed compensations written to `_compensation_failures` list. Background task retries 3x with exponential backoff. After 3 failures, alert via log CRITICAL.
- **Test**: `test_saga_retries_compensation` — failed compensation retried 3 times
- **Test**: `test_saga_dead_letter_on_failure` — 3x failure logged as CRITICAL

#### ATOM-P3-23: Adversarial embedding detection
- **File**: `src/ww/storage/qdrant_store.py` or new `src/ww/core/embedding_guard.py`
- **Root cause**: No detection of adversarial query/storage embeddings (M-46)
- **Fix**: Add norm validation (reject if L2 norm > 10x or < 0.01x of mean stored norm). Track running mean/std of stored embedding norms. Warn on outliers > 3 sigma.
- **Test**: `test_adversarial_embedding_detected` — 1000x norm vector rejected

#### ATOM-P3-24: Glymphatic datetime error handling
- **File**: `src/ww/nca/glymphatic_consolidation_bridge.py:502`
- **Root cause**: `datetime.fromisoformat()` with no try/except (M-52)
- **Fix**: Wrap in try/except, reject invalid timestamps with ValidationError.
- **Test**: `test_glymphatic_handles_bad_timestamp` — malformed string raises ValidationError, not crash

#### ATOM-P3-25: Feedback loop exploitation prevention
- **File**: `src/ww/api/routes/episodes.py:483-532`
- **Root cause**: Repeated fake positive feedback shifts embeddings to match common queries (L-5.2, elevated to P3)
- **Fix**: Rate-limit feedback: max 5 per memory_id per hour. Cap total embedding shift per feedback event to 0.05 cosine distance.
- **Test**: `test_feedback_rate_limited` — 6th feedback in 1 hour rejected
- **Test**: `test_feedback_shift_capped` — embedding moves at most 0.05 per event

### Striatal/Spatial/Energy/Neural Field (14 atoms)

#### ATOM-P3-26: Striatal lateral inhibition asymmetry
- **File**: `src/ww/nca/striatal_msn.py`
- **Root cause**: D1↔D2 lateral inhibition symmetric — should be asymmetric per Taverna 2008 (M-47)
- **Fix**: Add `d2_to_d1_strength: float = 1.3`, `d1_to_d2_strength: float = 0.7` config. D2→D1 inhibition 1.3x stronger.
- **Test**: `test_lateral_inhibition_asymmetric` — D2→D1 effect > D1→D2 effect

#### ATOM-P3-27: Hill function K_d validation
- **File**: `src/ww/nca/striatal_msn.py`
- **Root cause**: K_d=0 causes division by zero in Hill function (M-63)
- **Fix**: Add `assert K_d > 0` in config `__post_init__`. Add epsilon floor: `K_d = max(K_d, 1e-10)`.
- **Test**: `test_hill_function_kd_positive` — K_d=0 raises ValueError

#### ATOM-P3-28: Connectome weight parameterization
- **File**: `src/ww/nca/connectome.py`
- **Root cause**: Hardcoded weights (0.8, 0.9, etc.) with no literature citations (M-48)
- **Fix**: Add `weight_source: str` annotation to each connection. Create `CONNECTOME_WEIGHTS` dict with citations. Add `from_literature: bool` flag per connection. Document which are estimated vs cited.
- **Test**: `test_connectome_weights_documented` — every connection has weight_source annotation

#### ATOM-P3-29: Grid cell boundary effects
- **File**: `src/ww/nca/spatial_cells.py`
- **Root cause**: No boundary handling — edge artifacts in grid cell firing fields (M-49)
- **Fix**: Add `boundary_mode: str = "wrap"` config (options: "wrap" for toroidal, "clamp" for bounded). Toroidal avoids edge effects.
- **Test**: `test_grid_cell_toroidal_no_edge_artifacts` — firing rate uniform across arena edges

#### ATOM-P3-30: Hessian adaptive epsilon
- **File**: `src/ww/nca/stability.py`
- **Root cause**: Fixed epsilon=1e-5 causes catastrophic cancellation for large parameter values (M-56)
- **Fix**: Use relative epsilon: `eps = 1e-5 * max(1.0, abs(x_i))` per parameter dimension.
- **Test**: `test_hessian_relative_epsilon` — large parameter (1e6) produces accurate eigenvalues

#### ATOM-P3-31: Neural field operator splitting order
- **File**: `src/ww/nca/neural_field.py`
- **Root cause**: Naive sequential application of operators — Strang splitting improves accuracy (M-57)
- **Fix**: Implement Strang splitting: half-step diffusion → full-step reaction → half-step diffusion. Add `splitting_method: str = "strang"` config.
- **Test**: `test_strang_splitting_more_accurate` — compare error vs analytic solution, Strang < naive

#### ATOM-P3-32: CFL condition for reaction terms
- **File**: `src/ww/nca/neural_field.py`
- **Root cause**: CFL only checks diffusion stability, not reaction terms (M-58)
- **Fix**: Add reaction-term stability check: estimate max eigenvalue of reaction Jacobian, require `dt * max_eigenvalue < 1.0`. Reduce dt if violated.
- **Test**: `test_cfl_reaction_check` — stiff reaction term triggers dt reduction

#### ATOM-P3-33: Astrocyte Michaelis-Menten parameter documentation
- **File**: `src/ww/nca/astrocyte.py`
- **Root cause**: Fixed Km/Vmax with no temperature/pH dependence, excitotoxicity threshold undocumented (M-50)
- **Fix**: Add docstring with literature citations for Km/Vmax values. Add `excitotoxicity_threshold` config parameter (not hardcoded). Document that temperature/pH dependence is a planned enhancement.
- **Test**: `test_excitotoxicity_threshold_configurable` — threshold changeable via config

#### ATOM-P3-34: Glutamate Mg2+ voltage block improvement
- **File**: `src/ww/nca/glutamate_signaling.py`
- **Root cause**: Simplified sigmoid for Mg2+ block instead of Bhatt et al. (1998) exponential model (M-62)
- **Fix**: Replace sigmoid with `block = 1 / (1 + (Mg_ext / 3.57) * exp(-0.062 * V_m))` per Bhatt et al. Add `mg_block_model: str = "bhatt1998"` config.
- **Test**: `test_mg_block_bhatt_model` — at resting potential (-70mV), block >90%; at 0mV, block <30%

#### ATOM-P3-35: PAC phase bin configurability
- **File**: `src/ww/nca/oscillators.py`
- **Root cause**: Phase binning (18 bins) hardcoded — sensitivity undocumented (M-59)
- **Fix**: Add `pac_n_bins: int = 18` config. Add docstring noting Tort et al. 2010 recommendation of 18 bins.
- **Test**: `test_pac_bins_configurable` — PAC computed with 12 bins produces valid MI

#### ATOM-P3-36: WM capacity dynamic adjustment
- **File**: `src/ww/nca/wm_gating.py`
- **Root cause**: Fixed WM capacity (M-60)
- **Fix**: Add `dynamic_capacity: bool = False` config. When True, capacity = `base_capacity * arousal_factor` where `arousal_factor = gaussian(NE, optimal=0.6)`. Clamp to [3, 9].
- **Test**: `test_wm_dynamic_capacity` — high NE reduces WM capacity from 7 to ~5

#### ATOM-P3-37: Unified attention dynamic alpha
- **File**: `src/ww/nca/unified_attention.py`
- **Root cause**: Fixed alpha weight for capsule-transformer fusion (M-61)
- **Fix**: Add `adaptive_alpha: bool = False` config. When True, alpha = `sigmoid(learned_weight * agreement_score)`. Initialize `learned_weight=0.0` (equal weighting).
- **Test**: `test_attention_alpha_adaptive` — alpha changes with agreement score

#### ATOM-P3-38: Energy landscape Langevin adaptive stepping
- **File**: `src/ww/nca/energy.py`
- **Root cause**: Fixed step size in Langevin dynamics — biased samples for large dt (M-54)
- **Fix**: Add Metropolis-Hastings acceptance step. If energy change > expected for dt, reject sample. Add `use_mh_correction: bool = True` config.
- **Test**: `test_langevin_mh_correction` — distribution closer to target with correction than without

#### ATOM-P3-39: O(n^2) capsule routing vectorization
- **File**: `src/ww/nca/capsules.py:586-596`
- **Root cause**: Nested for-loops over `num_lower * num_output` (M-35)
- **Fix**: Replace nested loops with `np.einsum` or batched matrix operations.
- **Test**: `test_capsule_routing_performance` — 128 capsules, routing in <100ms (vs >1s with loops)

---

## Phase 5: P4 Hardening — LOW Findings (13 atoms)

### ATOM-P4-1: Place cell sigma scaling
- **File**: `src/ww/nca/spatial_cells.py`
- **Root cause**: Place cell Gaussian sigma=0.15 fixed — should scale with environment (O'Keefe & Burgess 1996)
- **Fix**: `sigma = base_sigma * sqrt(arena_size / reference_size)`. Default reference_size=1.0.
- **Test**: `test_place_cell_sigma_scales` — larger arena produces wider place fields

### ATOM-P4-2: TAN pause duration configurable
- **File**: `src/ww/nca/striatal_msn.py`
- **Fix**: Add `tan_pause_ms: float = 200.0` config (currently hardcoded).
- **Test**: `test_tan_pause_configurable` — different config produces different pause

### ATOM-P4-3: Habit formation saturation curve
- **File**: `src/ww/nca/striatal_msn.py`
- **Root cause**: Linear accumulation — can drift unboundedly
- **Fix**: Replace linear with sigmoid: `strength = max_strength * sigmoid(raw_strength / scale)`.
- **Test**: `test_habit_saturates` — 10000 updates, strength < max_strength

### ATOM-P4-4: Subiculum seeded RNG
- **File**: `src/ww/nca/hippocampus.py:476`
- **Root cause**: Uses global `np.random.randn` instead of `default_rng`
- **Fix**: Accept `seed` parameter, use `np.random.default_rng(seed)`.
- **Test**: `test_subiculum_reproducible` — same seed, same projection matrix

### ATOM-P4-5: Softmax overflow protection in Hopfield
- **File**: `src/ww/nca/energy.py`
- **Root cause**: `beta * X^T * query` may overflow — needs log-sum-exp trick
- **Fix**: `logits -= logits.max()` before `exp()`.
- **Test**: `test_hopfield_no_overflow` — beta=100, high-similarity patterns, no Inf

### ATOM-P4-6: FF sigmoid overflow protection
- **File**: `src/ww/nca/forward_forward.py:399`
- **Root cause**: `np.exp(-confidence)` overflows for very negative confidence
- **Fix**: Use `scipy.special.expit(confidence)` or clip input to [-500, 500].
- **Test**: `test_ff_classify_no_overflow` — confidence=-1000, no Inf

### ATOM-P4-7: FF-NCA goodness log-of-negative protection
- **File**: `src/ww/nca/forward_forward_nca_coupling.py`
- **Root cause**: `np.log(goodness + 1e-8)` — if goodness negative, log produces NaN
- **Fix**: `np.log(max(goodness, 1e-8))`.
- **Test**: `test_goodness_energy_no_nan` — negative goodness produces valid energy

### ATOM-P4-8: WM activation decay configurable
- **File**: `src/ww/nca/theta_gamma_integration.py:157`
- **Root cause**: 0.9 decay factor hardcoded
- **Fix**: Add `wm_decay: float = 0.9` config.
- **Test**: `test_wm_decay_configurable` — 0.5 decay empties slots faster

### ATOM-P4-9: Session isolation verification
- **File**: `src/ww/api/routes/episodes.py`
- **Root cause**: X-Session-ID header spoofable (L-5.3)
- **Fix**: Sessions tied to API key. Verify session ownership: `session.api_key == request.api_key`.
- **Test**: `test_session_isolation` — different API key cannot access other's session

### ATOM-P4-10: LC phasic seedable RNG
- **File**: `src/ww/nca/locus_coeruleus.py:724`
- **Root cause**: `np.random.random()` non-seedable
- **Fix**: Use `self._rng.random()` (already partially covered by P2-11, this ensures LC specifically).
- **Test**: `test_lc_phasic_reproducible` — same seed, same phasic decisions

### ATOM-P4-11: LC dead PFC gain attribute
- **File**: `src/ww/nca/locus_coeruleus.py:809-815`
- **Root cause**: `setattr` adds `pfc_gain` attribute never used in `trigger_phasic`
- **Fix**: Remove dead `setattr` block. Add TODO comment if PFC modulation is planned.
- **Test**: `test_lc_no_dead_attributes` — after `receive_pfc_modulation`, no `pfc_gain` attr on state

### ATOM-P4-12: Lyapunov exponent estimation note
- **File**: `src/ww/nca/stability.py`
- **Root cause**: Simplified trajectory divergence, not full Benettin algorithm
- **Fix**: Add docstring: "Simplified Lyapunov estimation via trajectory divergence. For publication-grade spectra, implement Benettin et al. 1980."
- **Test**: N/A (documentation-only)

### ATOM-P4-13: Cross-frequency coupling beyond PAC
- **File**: `src/ww/nca/oscillators.py`
- **Root cause**: Only PAC implemented, missing phase-phase and amplitude-amplitude coupling
- **Fix**: Add `:::planned` note to diagram. Add stub methods `compute_ppc()` and `compute_aac()` with `NotImplementedError`.
- **Test**: `test_ppc_aac_raise_not_implemented` — calling stubs raises cleanly

---

## Phase 6: Diagram Sync (7 atoms for remaining mismatches)

### ATOM-D1: DA decay tau — update diagram
- **File**: `docs/diagrams/33_state_neuromod.mmd`
- **Fix**: Replace `tau=2s` / `tau=5s` with `tau=0.2s (synaptic)` + note about PFC volume transmission
- **Depends on**: ATOM-P1-3 decision

### ATOM-D2: NE hyperarousal threshold — already covered by ATOM-P1-10

### ATOM-D3: ACh mode thresholds — already covered by ATOM-P1-11

### ATOM-D4: Orchestra Coordinator
- **File**: `docs/diagrams/13_neuromodulation_subsystem.mmd`
- **Fix**: Add `:::planned` to Orchestra Coordinator node. Add note: "Not yet implemented — individual modulators operate independently"
- **Depends on**: None

### ATOM-D5: VTA NAc GABA feedback
- **File**: `docs/diagrams/vta_circuit.mermaid`
- **Fix**: Add `:::planned` to NAc feedback and GABA/Glu neuron subtype annotations
- **Depends on**: None

### ATOM-D6: Spindle-SWR nesting
- **File**: `docs/diagrams/swr_replay.mermaid` or `sleep_cycle.mermaid`
- **Fix**: Add `:::planned` to spindle-SWR nesting timing. Add note: "Only spindle-delta coupling implemented"
- **Depends on**: None

### ATOM-D7: NE encoding gain
- **File**: `docs/diagrams/hippocampal_circuit.mermaid`
- **Fix**: After ATOM-P1-6 wires the code, update annotation from `:::partial` to solid
- **Depends on**: ATOM-P1-6

---

## Execution Dependencies

```
FOUNDATION-1 ──┬── P0-3, P0-5, P0-7, P0-12, P0-13, P1-7, P2-1, P2-14, P2-24
               │
FOUNDATION-2 ──┬── P0-4, P0-6, P2-2, P2-20, P2-25
               │
FOUNDATION-3 ──┬── P0-3, P0-4, P0-5, P0-7, P0-12, P0-13, P0-14
               │
All P0 atoms: Independent of each other (after foundations)
All P1 atoms: Independent of each other
P2 atoms: Independent (some reference foundations)
P3 atoms: Independent (some reference P0/P1 patterns)
P4 atoms: Independent
D atoms: After corresponding P1 atoms
```

---

## Execution Summary

| Phase | Atoms | Files Touched | Description |
|-------|-------|---------------|-------------|
| Foundations | 3 | 3 new files | validation.py, provenance.py, access_control.py |
| P0: Security | 14 | ~15 files | Eliminate all 9 CRITICAL + all HIGH security vulnerabilities |
| P1: Correctness | 16 | ~12 files | Fix wrong biology, numerical stability, dead code, spindle-ripple coupling |
| P2: Best Practice | 26 | ~20 files | Invariants, rate limits, provenance, reproducibility, encapsulation |
| P3: MEDIUM Hardening | 39 | ~25 files | Homeostatic mechanisms, overflow protection, parameter validation, biological accuracy |
| P4: LOW Hardening | 13 | ~10 files | Configurability, documentation, numerical edge cases, reproducibility |
| D: Diagrams | 7 | ~5 files | Remaining diagram-code mismatches |
| **Total** | **118** | **~50 unique** | **100% audit coverage** |

---

## Coverage Verification

| Severity | Findings | Atoms Covering | Coverage |
|----------|----------|----------------|----------|
| CRITICAL | 9 | P0-1 through P0-11 | **100%** |
| HIGH | 33 | P0-12 through P0-14, P1-13/14, P2-20 through P2-26, plus original P0/P1/P2 | **100%** |
| MEDIUM | 64 | P3-1 through P3-39, plus P2 atoms | **100%** |
| LOW | 39 | P4-1 through P4-13, plus P2-11 (RNG), documentation atoms | **100%** |
| Diagrams | 7 | D1 through D7 | **100%** |

**Every finding in the audit has a corresponding atom. No gaps.**

---

## Test Strategy

Every atom has 1-3 test assertions. Total: ~250 new tests.

```
tests/
├── core/
│   ├── test_validation.py           ← FOUNDATION-1
│   ├── test_provenance.py           ← FOUNDATION-2
│   └── test_access_control.py       ← FOUNDATION-3
├── security/
│   ├── test_pickle_rce.py           ← P0-1
│   ├── test_memory_injection.py     ← P0-3, P0-5, P0-6, P0-12, P0-13
│   ├── test_neuromod_manipulation.py ← P0-2, P0-4, P0-14
│   ├── test_learning_poisoning.py   ← P0-7
│   └── test_persistence_integrity.py ← P0-8, P0-9, P0-10, P0-11
├── correctness/
│   ├── test_biology_fixes.py        ← P1-1 through P1-14
│   ├── test_invariants.py           ← P2-3
│   └── test_numerical_stability.py  ← P3-1, P3-27, P3-30, P4-5, P4-6, P4-7
├── bestpractice/
│   ├── test_rate_limiting.py        ← P2-10, P3-8, P3-15, P3-21, P3-25
│   ├── test_reproducibility.py      ← P2-11, P2-19, P3-19, P4-4, P4-10
│   ├── test_drift_detection.py      ← P2-16, P2-22
│   ├── test_encapsulation.py        ← P2-5, P2-23, P3-2, P3-14, P3-17
│   └── test_provenance_chain.py     ← P2-2, P2-20, P2-25
├── hardening/
│   ├── test_homeostatic.py          ← P3-9, P3-10, P3-12
│   ├── test_biological_accuracy.py  ← P3-11, P3-13, P3-26, P3-29, P3-34
│   ├── test_performance.py          ← P3-39, P2-18
│   └── test_configurability.py      ← P3-33, P3-35, P3-36, P3-37, P4-2, P4-8
└── low_priority/
    ├── test_numerical_edge_cases.py ← P4-5, P4-6, P4-7
    └── test_documentation.py        ← P3-5, P3-7, P4-12 (verify docstrings exist)
```

Each test is a standalone assertion that can run independently. No test depends on another test. All tests are deterministic (seedable RNG where randomness is needed).
