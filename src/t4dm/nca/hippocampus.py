"""
Hippocampal Circuit for T4DM.

Sprint 1.1 (P0): Implements distinct DG/CA3/CA1 subregions with proper
information flow as identified in the CompBio architecture audit.

Biological Basis:
- Entorhinal Cortex (EC) -> Dentate Gyrus (DG): Pattern separation
- DG -> CA3: Sparse, orthogonalized representations
- CA3 recurrent: Autoassociation/pattern completion
- CA3 -> CA1: Novelty detection (comparing with EC input)
- CA1 -> EC: Output back to cortex

Key Functions:
- DG: Orthogonalize similar inputs (expansion recoding, sparse coding)
- CA3: Pattern completion via Modern Hopfield dynamics
- CA1: Novelty/mismatch detection (EC vs CA3 comparison)

Information Flow:
    EC input -> DG (separate) -> CA3 (complete) -> CA1 (compare with EC)
                                                         |
                                                         v
                                               novelty signal + output

References:
- Rolls (2013): The mechanisms for pattern completion and pattern separation
- Ramsauer et al. (2020): Hopfield Networks is All You Need
- Treves & Rolls (1994): Computational constraints on hippocampal function
- O'Reilly & McClelland (1994): Hippocampal conjunctive encoding
- Lisman & Grace (2005): Hippocampal-VTA loop and novelty-dependent memory
- Hasselmo et al. (2002): Cholinergic modulation of encoding vs retrieval
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

import numpy as np

from t4dm.core.access_control import CallerToken, require_capability
from t4dm.core.validation import validate_array

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class HippocampalMode(Enum):
    """Operating mode of hippocampal circuit."""
    ENCODING = "encoding"      # Store new pattern
    RETRIEVAL = "retrieval"    # Recall existing pattern
    AUTOMATIC = "automatic"    # Decide based on novelty


@dataclass
class HippocampalConfig:
    """Configuration for hippocampal circuit."""

    # Dimensions
    ec_dim: int = 1024       # Entorhinal cortex (input) dimension
    dg_dim: int = 4096       # Dentate gyrus (expanded) dimension
    ca3_dim: int = 1024      # CA3 (autoassociative) dimension
    ca1_dim: int = 1024      # CA1 (output) dimension
    subiculum_dim: int = 1024  # Subiculum (output to EC) dimension

    # DG parameters (pattern separation)
    dg_sparsity: float = 0.04        # ~4% activation (Jung & McNaughton 1993)
    dg_separation_threshold: float = 0.55  # Similarity threshold for separation
    dg_max_separation: float = 0.3   # Maximum separation magnitude
    dg_use_expansion: bool = True    # Expand to higher dim before sparsifying

    # CA3 parameters (pattern completion)
    ca3_beta: float = 8.0            # Hopfield inverse temperature
    ca3_max_patterns: int = 1000     # Maximum stored patterns
    ca3_max_iterations: int = 10     # Convergence iterations
    ca3_convergence_threshold: float = 0.001

    # CA1 parameters (novelty detection)
    ca1_novelty_threshold: float = 0.3   # Mismatch threshold for novelty
    ca1_encoding_threshold: float = 0.5  # High novelty -> encoding mode

    # Subiculum parameters (output modulation)
    subiculum_gain: float = 1.0  # Output gain modulation

    # Learning rates
    learning_rate: float = 0.01
    hebbian_decay: float = 0.999


@dataclass
class HippocampalState:
    """State of hippocampal circuit after processing."""

    # Pattern at each stage
    ec_input: np.ndarray
    dg_output: np.ndarray
    ca3_output: np.ndarray
    ca1_output: np.ndarray
    subiculum_output: np.ndarray  # Final output via subiculum

    # Metrics
    novelty_score: float           # CA1 mismatch (0=familiar, 1=novel)
    separation_magnitude: float     # How much DG separated the input
    completion_iterations: int      # CA3 settling steps
    completion_energy: float        # CA3 Hopfield energy

    # Mode decision
    mode: HippocampalMode
    pattern_id: UUID | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_novel(self) -> bool:
        """Whether input was classified as novel."""
        return self.mode == HippocampalMode.ENCODING

    @property
    def is_familiar(self) -> bool:
        """Whether input was classified as familiar."""
        return self.mode == HippocampalMode.RETRIEVAL


# =============================================================================
# Dentate Gyrus (Pattern Separation)
# =============================================================================


class DentateGyrusLayer:
    """
    Dentate Gyrus: Pattern separation via expansion and sparse coding.

    The DG receives dense input from EC and produces sparse, orthogonalized
    representations that reduce interference between similar patterns.

    Key mechanisms:
    1. Expansion: EC (1024) -> DG (4096) via random projection
    2. Sparsification: Keep only top ~4% of activations
    3. Orthogonalization: Project away from similar stored patterns

    This is critical for one-shot learning - preventing new memories
    from overwriting similar existing ones.
    """

    def __init__(
        self,
        config: HippocampalConfig,
        random_seed: int | None = None
    ):
        self.config = config
        self._rng = np.random.default_rng(random_seed)

        # Expansion weights (EC -> DG)
        # Xavier initialization for stable gradients
        scale = np.sqrt(2.0 / (config.ec_dim + config.dg_dim))
        self._expansion_weights = self._rng.normal(
            0, scale,
            size=(config.ec_dim, config.dg_dim)
        ).astype(np.float32)

        # Compression weights (DG -> CA3)
        scale = np.sqrt(2.0 / (config.dg_dim + config.ca3_dim))
        self._compression_weights = self._rng.normal(
            0, scale,
            size=(config.dg_dim, config.ca3_dim)
        ).astype(np.float32)

        # ATOM-P3-4: Recent patterns for separation comparison (bounded deque)
        self._recent_patterns: deque[np.ndarray] = deque(maxlen=100)

        logger.info(
            f"DentateGyrusLayer initialized: EC({config.ec_dim}) -> "
            f"DG({config.dg_dim}) -> CA3({config.ca3_dim}), "
            f"sparsity={config.dg_sparsity:.1%}"
        )

    def process(
        self,
        ec_input: np.ndarray,
        apply_separation: bool = True
    ) -> tuple[np.ndarray, float]:
        """
        Process input through dentate gyrus.

        Args:
            ec_input: Input from entorhinal cortex [ec_dim]
            apply_separation: Whether to orthogonalize against recent patterns

        Returns:
            (dg_output for CA3 [ca3_dim], separation_magnitude)
        """
        ec_input = np.asarray(ec_input, dtype=np.float32)

        # Normalize input
        norm = np.linalg.norm(ec_input)
        if norm > 0:
            ec_input = ec_input / norm

        # Step 1: Expand to DG dimension
        if self.config.dg_use_expansion:
            expanded = ec_input @ self._expansion_weights  # [dg_dim]
            # ReLU-like activation
            expanded = np.maximum(0, expanded)
        else:
            # Pad with zeros if not using expansion
            expanded = np.zeros(self.config.dg_dim, dtype=np.float32)
            expanded[:len(ec_input)] = ec_input

        # Step 2: Sparsify (keep top k%)
        sparse = self._sparsify(expanded)

        # Step 3: Orthogonalize against similar recent patterns
        separation_magnitude = 0.0
        if apply_separation and self._recent_patterns:
            sparse, separation_magnitude = self._orthogonalize(sparse)

        # Step 4: Compress to CA3 dimension
        ca3_input = sparse @ self._compression_weights  # [ca3_dim]

        # Normalize output
        out_norm = np.linalg.norm(ca3_input)
        if out_norm > 0:
            ca3_input = ca3_input / out_norm

        # Store for future separation comparisons (deque auto-evicts oldest)
        self._recent_patterns.append(sparse.copy())

        return ca3_input, separation_magnitude

    def _sparsify(self, pattern: np.ndarray) -> np.ndarray:
        """Apply sparse coding (keep top k% of activations)."""
        k = max(1, int(len(pattern) * self.config.dg_sparsity))

        # ATOM-P2-17: k-WTA sparsification (replaces soft-shrinkage)
        # Guarantees exactly k nonzero elements for exact sparsity control
        if k < len(pattern):
            # Find indices of top k elements by absolute value
            topk_idx = np.argpartition(np.abs(pattern), -k)[-k:]
            sparse = np.zeros_like(pattern)
            sparse[topk_idx] = pattern[topk_idx]
        else:
            sparse = pattern.copy()

        return sparse

    def _orthogonalize(
        self,
        pattern: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Orthogonalize pattern away from similar recent patterns."""
        if not self._recent_patterns:
            return pattern, 0.0

        # Find similar patterns
        similarities = []
        for stored in list(self._recent_patterns)[-50:]:  # Check recent 50
            norm_s = np.linalg.norm(stored)
            norm_p = np.linalg.norm(pattern)
            if norm_s > 0 and norm_p > 0:
                sim = np.dot(pattern, stored) / (norm_p * norm_s)
                similarities.append((sim, stored))

        # Filter by threshold
        similar = [(s, p) for s, p in similarities
                   if s > self.config.dg_separation_threshold]

        if not similar:
            return pattern, 0.0

        # Compute interference direction (weighted centroid)
        weights = np.array([s for s, _ in similar])
        weights = weights / weights.sum()
        vectors = np.array([p for _, p in similar])
        centroid = np.average(vectors, axis=0, weights=weights)

        # Normalize centroid
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-8:
            return pattern, 0.0
        centroid = centroid / centroid_norm

        # Compute separation strength based on max similarity
        max_sim = max(s for s, _ in similar)
        sep_strength = min(
            self.config.dg_max_separation,
            (max_sim - self.config.dg_separation_threshold) /
            (1.0 - self.config.dg_separation_threshold) *
            self.config.dg_max_separation
        )

        # Project away from centroid
        projection = np.dot(pattern, centroid) * centroid
        orthogonalized = pattern - sep_strength * projection

        # Add small random perturbation (expansion recoding)
        noise = self._rng.normal(0, 0.01 * sep_strength, size=pattern.shape)
        orthogonalized = orthogonalized + noise

        separation_magnitude = float(np.linalg.norm(orthogonalized - pattern))
        return orthogonalized.astype(np.float32), separation_magnitude

    def clear_recent(self) -> None:
        """Clear recent pattern history."""
        self._recent_patterns.clear()


# =============================================================================
# CA3 (Pattern Completion)
# =============================================================================


class CA3Layer:
    """
    CA3: Autoassociative pattern completion via Modern Hopfield dynamics.

    CA3 has extensive recurrent connections, making it ideal for pattern
    completion. Given a partial or noisy cue, it reconstructs the full
    stored pattern through attractor dynamics.

    Uses Modern Hopfield networks (Ramsauer et al. 2020) for exponential
    storage capacity.
    """

    def __init__(self, config: HippocampalConfig):
        self.config = config

        # Stored patterns
        self._patterns: list[np.ndarray] = []
        self._pattern_ids: list[UUID] = []
        self._pattern_metadata: list[dict] = []

        # ATOM-P2-2: Provenance tracking for stored patterns
        self._provenance_records: dict = {}

        logger.info(
            f"CA3Layer initialized: dim={config.ca3_dim}, "
            f"beta={config.ca3_beta}, max_patterns={config.ca3_max_patterns}"
        )

    def store(
        self,
        pattern: np.ndarray,
        metadata: dict | None = None,
        provenance: dict | None = None
    ) -> UUID:
        """
        Store a pattern in CA3.

        Args:
            pattern: Pattern to store [ca3_dim]
            metadata: Optional metadata (e.g., episode_id, timestamp)
            provenance: Optional provenance tracking (ATOM-P2-2)

        Returns:
            UUID for the stored pattern
        """
        pattern = np.asarray(pattern, dtype=np.float32)

        # Normalize
        norm = np.linalg.norm(pattern)
        if norm > 0:
            pattern = pattern / norm

        pattern_id = uuid4()
        self._patterns.append(pattern)
        self._pattern_ids.append(pattern_id)
        self._pattern_metadata.append(metadata or {})

        # ATOM-P2-2: Store provenance alongside pattern
        if provenance is not None:
            self._provenance_records[len(self._patterns) - 1] = provenance

        # Enforce capacity limit (FIFO eviction)
        if len(self._patterns) > self.config.ca3_max_patterns:
            self._patterns.pop(0)
            evicted_id = self._pattern_ids.pop(0)
            self._pattern_metadata.pop(0)
            # Also clean up provenance for evicted pattern
            if 0 in self._provenance_records:
                del self._provenance_records[0]
            # Shift provenance indices after eviction
            self._provenance_records = {k-1: v for k, v in self._provenance_records.items() if k > 0}

        return pattern_id

    def complete(
        self,
        query: np.ndarray
    ) -> tuple[np.ndarray, int, float, UUID | None]:
        """
        Complete a pattern using Modern Hopfield dynamics.

        Args:
            query: Partial or noisy pattern [ca3_dim]

        Returns:
            (completed_pattern, iterations, energy, best_match_id)
        """
        # ATOM-P3-3: Return None for empty store instead of query
        if not self._patterns:
            return None, 0, 0.0, None

        query = np.asarray(query, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        patterns = np.array(self._patterns)
        # ATOM-P3-2: Beta is internal only (removed from public signature)
        effective_beta = self.config.ca3_beta

        current = query.copy()
        converged = False

        for iteration in range(self.config.ca3_max_iterations):
            # Modern Hopfield update: softmax(beta * sim) @ patterns
            similarities = patterns @ current
            scaled = effective_beta * similarities
            # ATOM-P3-1: Log-sum-exp trick for numerical stability
            max_sim = np.max(scaled)
            attention = np.exp(scaled - max_sim)
            attention = attention / attention.sum()

            new_pattern = attention @ patterns

            # Normalize
            new_norm = np.linalg.norm(new_pattern)
            if new_norm > 0:
                new_pattern = new_pattern / new_norm

            # Check convergence
            delta = np.linalg.norm(new_pattern - current)
            current = new_pattern

            if delta < self.config.ca3_convergence_threshold:
                converged = True
                break

        # Compute energy
        similarities = patterns @ current
        max_sim = similarities.max()
        energy = -(np.log(np.sum(np.exp(effective_beta * (similarities - max_sim)))) +
                   effective_beta * max_sim) / effective_beta

        # Find best match
        best_idx = np.argmax(similarities)
        best_id = self._pattern_ids[best_idx]

        return current, iteration + 1, float(energy), best_id

    def get_pattern_count(self) -> int:
        """Get number of stored patterns."""
        return len(self._patterns)

    def clear(self) -> None:
        """Clear all stored patterns."""
        self._patterns.clear()
        self._pattern_ids.clear()
        self._pattern_metadata.clear()


# =============================================================================
# Subiculum (Output Modulation)
# =============================================================================


class SubiculumLayer:
    """
    Subiculum: Output stage between CA1 and EC.

    The subiculum receives CA1 output and applies gain modulation before
    sending signals back to entorhinal cortex, completing the hippocampal loop.

    Key functions:
    - Gain modulation of CA1 output
    - Final output stage to cortical areas
    - Completes hippocampal-cortical loop
    """

    def __init__(self, config: HippocampalConfig, seed: int | None = None):
        self.config = config

        # ATOM-P4-4: Seeded RNG for reproducible projection matrix
        rng = np.random.default_rng(seed)

        # Projection from CA1 to subiculum dimension (if needed)
        self._ca1_projection: np.ndarray | None = None
        if config.ca1_dim != config.subiculum_dim:
            scale = np.sqrt(2.0 / (config.ca1_dim + config.subiculum_dim))
            self._ca1_projection = rng.standard_normal(
                (config.ca1_dim, config.subiculum_dim)
            ).astype(np.float32) * scale

        logger.info(
            f"SubiculumLayer initialized: dim={config.subiculum_dim}, "
            f"gain={config.subiculum_gain}, seed={seed}"
        )

    def process(self, ca1_output: np.ndarray) -> np.ndarray:
        """
        Process CA1 output through subiculum.

        Args:
            ca1_output: Input from CA1 [ca1_dim]

        Returns:
            subiculum_output: Modulated output [subiculum_dim]
        """
        ca1_output = np.asarray(ca1_output, dtype=np.float32)

        # Project to subiculum dimension
        if self._ca1_projection is not None:
            output = ca1_output @ self._ca1_projection
        else:
            output = ca1_output.copy()

        # Apply gain modulation
        output = output * self.config.subiculum_gain

        # Normalize
        norm = np.linalg.norm(output)
        if norm > 0:
            output = output / norm

        return output.astype(np.float32)


# =============================================================================
# CA1 (Novelty Detection)
# =============================================================================


class CA1Layer:
    """
    CA1: Novelty detection by comparing CA3 output with EC input.

    CA1 receives input from both CA3 (via Schaffer collaterals) and EC
    (via temporoammonic pathway). By comparing these, CA1 can detect
    novelty/mismatch:

    - High match (EC ~ CA3): Familiar pattern, retrieval mode
    - Low match (EC != CA3): Novel pattern, encoding mode

    This drives the encoding vs retrieval mode decision.
    """

    def __init__(self, config: HippocampalConfig):
        self.config = config

        # Projection from EC to CA1 dimension (if needed)
        self._ec_projection: np.ndarray | None = None
        if config.ec_dim != config.ca1_dim:
            scale = np.sqrt(2.0 / (config.ec_dim + config.ca1_dim))
            self._ec_projection = np.random.randn(
                config.ec_dim, config.ca1_dim
            ).astype(np.float32) * scale

        # Projection from CA3 to CA1 dimension (if needed)
        self._ca3_projection: np.ndarray | None = None
        if config.ca3_dim != config.ca1_dim:
            scale = np.sqrt(2.0 / (config.ca3_dim + config.ca1_dim))
            self._ca3_projection = np.random.randn(
                config.ca3_dim, config.ca1_dim
            ).astype(np.float32) * scale

        logger.info(
            f"CA1Layer initialized: dim={config.ca1_dim}, "
            f"novelty_threshold={config.ca1_novelty_threshold}"
        )

    def process(
        self,
        ec_input: np.ndarray,
        ca3_output: np.ndarray
    ) -> tuple[np.ndarray, float, HippocampalMode]:
        """
        Process CA3 output and compute novelty.

        Args:
            ec_input: Original EC input [ec_dim]
            ca3_output: CA3 completed pattern [ca3_dim]

        Returns:
            (ca1_output, novelty_score, suggested_mode)
        """
        ec_input = np.asarray(ec_input, dtype=np.float32)
        ca3_output = np.asarray(ca3_output, dtype=np.float32)

        # Project to CA1 dimension
        if self._ec_projection is not None:
            ec_projected = ec_input @ self._ec_projection
        else:
            ec_projected = ec_input.copy()

        if self._ca3_projection is not None:
            ca3_projected = ca3_output @ self._ca3_projection
        else:
            ca3_projected = ca3_output.copy()

        # Normalize
        ec_norm = np.linalg.norm(ec_projected)
        ca3_norm = np.linalg.norm(ca3_projected)
        if ec_norm > 0:
            ec_projected = ec_projected / ec_norm
        if ca3_norm > 0:
            ca3_projected = ca3_projected / ca3_norm

        # Compute novelty as mismatch (1 - cosine similarity)
        similarity = np.dot(ec_projected, ca3_projected)
        # ATOM-P3-6: Explicit clamping of novelty score to [0, 1]
        novelty_score = float(np.clip(1.0 - similarity, 0.0, 1.0))

        # Combine EC and CA3 for output (CA1 integrates both)
        # Weight by familiarity: more familiar -> more CA3, more novel -> more EC
        alpha = 1.0 - novelty_score  # Familiarity weight
        ca1_output = alpha * ca3_projected + (1 - alpha) * ec_projected

        # Normalize output
        out_norm = np.linalg.norm(ca1_output)
        if out_norm > 0:
            ca1_output = ca1_output / out_norm

        # Determine mode based on novelty
        if novelty_score > self.config.ca1_encoding_threshold:
            mode = HippocampalMode.ENCODING
        elif novelty_score < self.config.ca1_novelty_threshold:
            mode = HippocampalMode.RETRIEVAL
        else:
            mode = HippocampalMode.AUTOMATIC  # Intermediate

        return ca1_output.astype(np.float32), float(novelty_score), mode


# =============================================================================
# Integrated Hippocampal Circuit
# =============================================================================


class HippocampalCircuit:
    """
    Complete hippocampal circuit integrating DG, CA3, and CA1.

    This is the Sprint 1.1 deliverable: a cohesive circuit that:
    1. Pattern separates input via DG
    2. Stores/retrieves patterns via CA3
    3. Detects novelty and decides encoding vs retrieval via CA1

    P1.3 Enhancement: Theta Phase Gating
    - Theta phase (4-8 Hz) gates encoding vs retrieval
    - Phase 0-π: Encoding favored (high plasticity, LTP)
    - Phase π-2π: Retrieval favored (pattern completion)

    Information flow:
        EC -> DG (separate) -> CA3 (store/complete) -> CA1 (compare with EC)
                                                            |
                                                            v
                                                   novelty + output
    """

    def __init__(
        self,
        config: HippocampalConfig | None = None,
        random_seed: int | None = None,
        oscillator: FrequencyBandGenerator | None = None
    ):
        self.config = config or HippocampalConfig()

        # Initialize layers
        self.dg = DentateGyrusLayer(self.config, random_seed)
        self.ca3 = CA3Layer(self.config)
        self.ca1 = CA1Layer(self.config)
        self.subiculum = SubiculumLayer(self.config)

        # P1.3: Theta phase gating
        self._oscillator = oscillator
        self._use_theta_gating = oscillator is not None
        self._theta_encoding_bias = 0.3  # Bias toward encoding at theta peak
        self._theta_retrieval_bias = 0.3  # Bias toward retrieval at theta trough

        # P1.4: VTA dopamine connection
        self._vta: VTACircuit | None = None
        self._novelty_to_vta_weight = 0.4  # How much novelty affects dopamine

        # C4: ACh/NE neuromodulator gating
        self._ach_level: float = 0.3  # Current acetylcholine level
        self._ne_level: float = 0.3   # Current norepinephrine level
        self._ach_encoding_threshold: float = 0.6  # High ACh = encoding mode
        self._ach_retrieval_threshold: float = 0.4  # Low ACh = retrieval mode
        self._ne_encoding_gain: float = 1.0  # NE multiplier on DG expansion

        # Processing history
        self._history: list[HippocampalState] = []
        self._max_history: int = 1000

        gating_msg = ", theta-gated" if self._use_theta_gating else ""
        logger.info(
            f"HippocampalCircuit initialized: "
            f"EC({self.config.ec_dim}) -> DG -> CA3 -> CA1 -> Subiculum({self.config.subiculum_dim}){gating_msg}"
        )

    def process(
        self,
        ec_input: np.ndarray,
        mode: HippocampalMode | None = None,
        store_if_novel: bool = True,
        metadata: dict | None = None,
        ec_layer2_input: np.ndarray | None = None,
        ec_layer3_input: np.ndarray | None = None
    ) -> HippocampalState:
        """
        Process input through the hippocampal circuit.

        Args:
            ec_input: Input from entorhinal cortex [ec_dim] (used for both layers if specific layers not provided)
            mode: Force encoding or retrieval (None = automatic)
            store_if_novel: Whether to store pattern if detected as novel
            metadata: Optional metadata to store with pattern
            ec_layer2_input: EC Layer II input (perforant path to DG/CA3). If None, uses ec_input.
            ec_layer3_input: EC Layer III input (temporoammonic path to CA1). If None, uses ec_input.

        Returns:
            HippocampalState with outputs and metrics
        """
        ec_input = np.asarray(ec_input, dtype=np.float32)

        # ATOM-P1-7: Reject zero-vector input to prevent degenerate DG state
        if np.linalg.norm(ec_input) < 1e-8:
            from t4dm.core.validation import ValidationError
            raise ValidationError("ec_input", "Zero-vector input to hippocampal circuit")

        # ATOM-P0-3: Validate input dimension at DG entry point
        validate_array(ec_input, expected_dim=self.config.ec_dim, name="ec_input")

        # Split EC input into Layer II and Layer III (backward compatible)
        if ec_layer2_input is None:
            ec_layer2_input = ec_input
        else:
            ec_layer2_input = np.asarray(ec_layer2_input, dtype=np.float32)

        if ec_layer3_input is None:
            ec_layer3_input = ec_input
        else:
            ec_layer3_input = np.asarray(ec_layer3_input, dtype=np.float32)

        # ATOM-P1-9: Normalize inputs without mutating caller's arrays
        norm = np.linalg.norm(ec_input)
        if norm > 0:
            ec_input = ec_input / norm

        norm = np.linalg.norm(ec_layer2_input)
        if norm > 0:
            ec_layer2_input = ec_layer2_input / norm

        norm = np.linalg.norm(ec_layer3_input)
        if norm > 0:
            ec_layer3_input = ec_layer3_input / norm

        # ATOM-P1-6: Apply NE encoding gain if NE is above threshold
        if hasattr(self, '_ne_level') and self._ne_level > 0.3:
            ec_layer2_input = ec_layer2_input * self._apply_ne_encoding_gain()

        # Step 1: DG pattern separation (EC Layer II input via perforant path)
        dg_output, separation_magnitude = self.dg.process(
            ec_layer2_input,
            apply_separation=True
        )

        # Step 2: CA3 pattern completion/retrieval
        ca3_output, iterations, energy, pattern_id = self.ca3.complete(
            dg_output
        )

        # ATOM-P3-3: Handle None return from empty CA3 store
        if ca3_output is None:
            # No stored patterns - use DG output directly
            ca3_output = dg_output

        # Step 2.5: Apply ACh gating to CA3→CA1 pathway (C4)
        # High ACh suppresses CA3→CA1 for encoding mode
        ca3_gated = self._apply_ach_gating(ca3_output, ec_layer3_input)

        # Step 3: CA1 novelty detection (EC Layer III input via temporoammonic pathway)
        ca1_output, novelty_score, detected_mode = self.ca1.process(
            ec_layer3_input,
            ca3_gated
        )

        # Step 4: Subiculum output modulation
        subiculum_output = self.subiculum.process(ca1_output)

        # P1.3: Apply theta phase gating to mode decision
        if mode is not None:
            final_mode = mode
        else:
            final_mode = self._apply_theta_gating(detected_mode, novelty_score)

        # Store if novel and requested
        stored_id = None
        if final_mode == HippocampalMode.ENCODING and store_if_novel:
            stored_id = self.ca3.store(dg_output, metadata)

        # Create state
        state = HippocampalState(
            ec_input=ec_input.copy(),
            dg_output=dg_output.copy(),
            ca3_output=ca3_output.copy(),
            ca1_output=ca1_output.copy(),
            subiculum_output=subiculum_output.copy(),
            novelty_score=novelty_score,
            separation_magnitude=separation_magnitude,
            completion_iterations=iterations,
            completion_energy=energy,
            mode=final_mode,
            pattern_id=stored_id or pattern_id
        )

        # P5.6: Auto-send novelty to VTA for dopamine modulation
        # Biological basis: Lisman & Grace (2005) - Hippocampal-VTA loop
        # Novel stimuli trigger VTA dopamine which enhances encoding
        da_level = None
        if self._vta is not None:
            try:
                da_level = self._send_novelty_to_vta(novelty_score)
                if da_level is not None:
                    logger.debug(
                        f"P5.6: Hippocampus→VTA novelty signal: "
                        f"novelty={novelty_score:.3f} → DA={da_level:.3f}"
                    )
            except Exception as e:
                logger.warning(f"VTA novelty signaling failed: {e}")

        # Store in history
        self._history.append(state)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        logger.debug(
            f"Hippocampal processing: mode={final_mode.value}, "
            f"novelty={novelty_score:.3f}, separation={separation_magnitude:.4f}"
            f"{f', DA={da_level:.3f}' if da_level is not None else ''}"
        )

        return state

    def encode(
        self,
        ec_input: np.ndarray,
        metadata: dict | None = None
    ) -> HippocampalState:
        """
        Force encoding mode (store new pattern).

        Args:
            ec_input: Input pattern
            metadata: Optional metadata

        Returns:
            HippocampalState
        """
        return self.process(
            ec_input,
            mode=HippocampalMode.ENCODING,
            store_if_novel=True,
            metadata=metadata
        )

    def retrieve(
        self,
        ec_input: np.ndarray
    ) -> HippocampalState:
        """
        Force retrieval mode (recall existing pattern).

        Args:
            ec_input: Query pattern

        Returns:
            HippocampalState
        """
        return self.process(
            ec_input,
            mode=HippocampalMode.RETRIEVAL,
            store_if_novel=False
        )

    def get_novelty_threshold(self) -> float:
        """Get current novelty threshold."""
        return self.config.ca1_novelty_threshold

    def set_novelty_threshold(self, threshold: float) -> None:
        """Set novelty threshold (affects encoding vs retrieval decision)."""
        self.config.ca1_novelty_threshold = max(0.0, min(1.0, threshold))

    def get_pattern_count(self) -> int:
        """Get number of stored patterns in CA3."""
        return self.ca3.get_pattern_count()

    def get_history(self, limit: int = 100) -> list[HippocampalState]:
        """Get recent processing history."""
        return self._history[-limit:]

    def get_stats(self) -> dict:
        """Get circuit statistics."""
        if not self._history:
            return {
                "total_processed": 0,
                "encoding_count": 0,
                "retrieval_count": 0,
                "avg_novelty": 0.0,
                "avg_separation": 0.0,
                "stored_patterns": self.ca3.get_pattern_count()
            }

        encodings = [s for s in self._history if s.is_novel]
        retrievals = [s for s in self._history if s.is_familiar]

        return {
            "total_processed": len(self._history),
            "encoding_count": len(encodings),
            "retrieval_count": len(retrievals),
            "encoding_rate": len(encodings) / len(self._history),
            "avg_novelty": float(np.mean([s.novelty_score for s in self._history])),
            "avg_separation": float(np.mean([s.separation_magnitude for s in self._history])),
            "avg_completion_iters": float(np.mean([s.completion_iterations for s in self._history])),
            "stored_patterns": self.ca3.get_pattern_count()
        }

    def clear(self) -> None:
        """Clear all stored patterns and history."""
        self.dg.clear_recent()
        self.ca3.clear()
        self._history.clear()

    # -------------------------------------------------------------------------
    # P1.3: Theta Phase Gating
    # -------------------------------------------------------------------------

    def _apply_theta_gating(
        self,
        detected_mode: HippocampalMode,
        novelty_score: float
    ) -> HippocampalMode:
        """
        Apply theta phase gating to mode decision.

        Biological basis:
        - Theta phase 0-π (encoding): LTP favored, new memories can form
        - Theta phase π-2π (retrieval): Pattern completion favored
        - Hasselmo et al. (2002): Cholinergic modulation of theta-dependent encoding

        The theta phase biases the encoding/retrieval decision:
        - At encoding phase: Lower threshold for encoding (novel patterns more likely stored)
        - At retrieval phase: Higher threshold (favor pattern completion)

        Args:
            detected_mode: Mode suggested by CA1 novelty detection
            novelty_score: CA1 novelty score (0=familiar, 1=novel)

        Returns:
            Final mode after theta gating
        """
        if not self._use_theta_gating or self._oscillator is None:
            return detected_mode

        # Get theta phase signals directly from theta oscillator
        phase = self._oscillator.theta.phase
        encoding_signal = 0.5 * (1.0 + np.cos(phase))  # 1.0 at phase 0, 0.0 at phase π
        retrieval_signal = 1.0 - encoding_signal  # Complementary

        # Adjust novelty threshold based on theta phase
        # At encoding phase: lower effective novelty threshold (encode more)
        # At retrieval phase: higher effective novelty threshold (retrieve more)
        phase_bias = (encoding_signal - 0.5) * 2  # Range: [-1, 1]

        # Compute effective novelty (shifted by phase)
        effective_novelty = novelty_score + phase_bias * self._theta_encoding_bias

        # Determine mode with phase-adjusted threshold
        if effective_novelty > self.config.ca1_encoding_threshold:
            return HippocampalMode.ENCODING
        elif effective_novelty < self.config.ca1_novelty_threshold:
            return HippocampalMode.RETRIEVAL
        else:
            # Ambiguous - use phase to break tie
            if encoding_signal > 0.5:
                return HippocampalMode.ENCODING
            else:
                return HippocampalMode.RETRIEVAL

    def set_oscillator(self, oscillator: FrequencyBandGenerator) -> None:
        """
        Set or update the theta oscillator for phase gating.

        Args:
            oscillator: FrequencyBandGenerator instance
        """
        self._oscillator = oscillator
        self._use_theta_gating = True
        logger.info("Theta phase gating enabled")

    def disable_theta_gating(self) -> None:
        """Disable theta phase gating."""
        self._use_theta_gating = False
        logger.info("Theta phase gating disabled")

    def get_encoding_probability(self) -> float:
        """
        Get current probability of encoding based on theta phase.

        Returns:
            Probability [0, 1] favoring encoding at current theta phase
        """
        if not self._use_theta_gating or self._oscillator is None:
            return 0.5  # Neutral when no oscillator
        # Compute from theta.phase directly (not cached state)
        phase = self._oscillator.theta.phase
        return 0.5 * (1.0 + np.cos(phase))

    def get_theta_phase(self) -> float | None:
        """
        Get current theta phase in radians.

        Returns:
            Phase [0, 2π] or None if no oscillator
        """
        if not self._use_theta_gating or self._oscillator is None:
            return None
        # Read from theta oscillator directly
        return self._oscillator.theta.phase

    # -------------------------------------------------------------------------
    # P1.4: VTA Dopamine Connection
    # -------------------------------------------------------------------------

    def connect_to_vta(self, vta: VTACircuit) -> None:
        """
        Connect hippocampus to VTA for novelty → dopamine signaling.

        Biological basis:
        - Novel stimuli are inherently salient and rewarding
        - CA1 mismatch detection triggers VTA dopamine release
        - This creates a learning signal for memory encoding

        Args:
            vta: VTACircuit instance to connect
        """
        self._vta = vta
        logger.info("Hippocampus connected to VTA")

    def disconnect_vta(self) -> None:
        """Disconnect from VTA."""
        self._vta = None
        logger.info("Hippocampus disconnected from VTA")

    def set_novelty_weight(self, weight: float) -> None:
        """
        Set weight for novelty → dopamine conversion.

        Args:
            weight: How strongly novelty drives dopamine [0, 1]
        """
        self._novelty_to_vta_weight = float(np.clip(weight, 0.0, 1.0))

    def _send_novelty_to_vta(self, novelty_score: float) -> float | None:
        """
        Send CA1 novelty signal to VTA.

        This converts novelty (mismatch) to a reward-like signal:
        - High novelty → positive RPE → dopamine burst
        - Low novelty → near-zero RPE → tonic dopamine

        Args:
            novelty_score: CA1 novelty score [0, 1]

        Returns:
            VTA dopamine level after update, or None if not connected
        """
        if self._vta is None:
            return None

        # Convert novelty to RPE-like signal
        # Novelty is centered at 0.5: above = surprising/novel, below = expected
        # This creates a bidirectional signal for VTA
        novelty_rpe = (novelty_score - 0.3) * self._novelty_to_vta_weight

        # Process through VTA
        da_level = self._vta.process_rpe(novelty_rpe)

        return da_level

    def get_last_dopamine(self) -> float | None:
        """
        Get the last dopamine level from connected VTA.

        Returns:
            Dopamine level [0, 1] or None if not connected
        """
        if self._vta is None:
            return None
        return self._vta.state.current_da

    def process_with_dopamine(
        self,
        ec_input: np.ndarray,
        mode: HippocampalMode | None = None,
        store_if_novel: bool = True,
        metadata: dict | None = None
    ) -> tuple[HippocampalState, float | None]:
        """
        Process input and return dopamine response.

        Combines hippocampal processing with VTA integration.

        Args:
            ec_input: Input from entorhinal cortex
            mode: Force encoding or retrieval
            store_if_novel: Store if detected as novel
            metadata: Optional metadata

        Returns:
            (HippocampalState, dopamine_level)
        """
        # Standard processing
        state = self.process(ec_input, mode, store_if_novel, metadata)

        # Send novelty to VTA
        da_level = self._send_novelty_to_vta(state.novelty_score)

        return state, da_level

    # -------------------------------------------------------------------------
    # C4: ACh/NE Neuromodulator Gating
    # -------------------------------------------------------------------------

    def receive_ach(self, level: float, token: CallerToken | None = None) -> None:
        """
        C4: Receive acetylcholine signal for encoding/retrieval gating.

        High ACh (>0.6): Suppress CA3→CA1 Schaffer pathway, enable EC→CA1
                        (encoding mode - new patterns stored)
        Low ACh (<0.4): Enable CA3→CA1 pattern completion (retrieval mode)

        Biological basis: Hasselmo et al. (2002) - cholinergic modulation
        of encoding vs retrieval dynamics in hippocampus.

        Args:
            level: Acetylcholine level [0, 1]
            token: Optional access control token (only enforced if provided)
        """
        if token is not None:
            require_capability(token, "set_neuromod")
        self._ach_level = float(np.clip(level, 0.0, 1.0))

    def receive_ne(self, level: float, token: CallerToken | None = None) -> None:
        """
        C4: Receive norepinephrine signal for arousal-dependent encoding.

        NE multiplies DG pattern separation gain. High NE (arousal) →
        stronger expansion/separation → better encoding of salient events.

        Biological basis: LC-NE enhances DG neurogenesis and pattern
        separation during arousal (Hagena et al. 2016).

        Args:
            level: Norepinephrine level [0, 1]
            token: Optional access control token (only enforced if provided)
        """
        if token is not None:
            require_capability(token, "set_neuromod")
        self._ne_level = float(np.clip(level, 0.0, 1.0))

    def _apply_ach_gating(self, ca3_output: np.ndarray, ec_input: np.ndarray) -> np.ndarray:
        """
        Apply ACh gating to CA3→CA1 vs EC→CA1 pathways.

        High ACh: Suppress CA3→CA1 (Schaffer collaterals), favor EC→CA1
        Low ACh: Enable CA3→CA1 pattern completion

        Args:
            ca3_output: CA3 output pattern
            ca3_output: EC input pattern

        Returns:
            Modulated CA3 output
        """
        if self._ach_level > self._ach_encoding_threshold:
            # High ACh: encoding mode - suppress CA3→CA1
            suppression = (self._ach_level - self._ach_encoding_threshold) / (1.0 - self._ach_encoding_threshold)
            suppression = np.clip(suppression, 0.0, 0.8)  # Max 80% suppression
            return ca3_output * (1.0 - suppression)
        else:
            # Normal or retrieval mode
            return ca3_output

    def _apply_ne_encoding_gain(self) -> float:
        """
        Compute NE-modulated encoding gain for DG expansion.

        Returns:
            Gain multiplier for DG pattern separation [1.0, 2.0]
        """
        # Linear gain: NE=0 → 1.0x, NE=1 → 2.0x
        gain = 1.0 + self._ne_level
        return float(np.clip(gain, 1.0, 2.0))


# =============================================================================
# Factory Functions
# =============================================================================


def create_hippocampal_circuit(
    ec_dim: int = 1024,
    dg_expansion_factor: int = 4,
    ca3_beta: float = 8.0,
    novelty_threshold: float = 0.3,
    **kwargs
) -> HippocampalCircuit:
    """
    Create a hippocampal circuit with common defaults.

    Args:
        ec_dim: Entorhinal cortex (input) dimension
        dg_expansion_factor: DG dimension = ec_dim * factor
        ca3_beta: Hopfield inverse temperature
        novelty_threshold: Threshold for novelty detection
        **kwargs: Additional config parameters

    Returns:
        Configured HippocampalCircuit
    """
    config = HippocampalConfig(
        ec_dim=ec_dim,
        dg_dim=ec_dim * dg_expansion_factor,
        ca3_dim=ec_dim,
        ca1_dim=ec_dim,
        ca3_beta=ca3_beta,
        ca1_novelty_threshold=novelty_threshold,
        **kwargs
    )
    return HippocampalCircuit(config)


# Backward compatibility aliases
HippocampalSystem = HippocampalCircuit
Hippocampus = HippocampalCircuit


__all__ = [
    "HippocampalConfig",
    "HippocampalState",
    "HippocampalMode",
    "HippocampalCircuit",
    "HippocampalSystem",  # Alias
    "Hippocampus",  # Alias
    "DentateGyrusLayer",
    "CA3Layer",
    "CA1Layer",
    "SubiculumLayer",
    "create_hippocampal_circuit",
]
