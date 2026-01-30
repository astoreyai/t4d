"""
LearnedMemoryGate - Online learning for memory storage decisions.

This module implements the algorithm designed in docs/LEARNED_MEMORY_GATE_DESIGN.md.
It uses online Bayesian logistic regression with Thompson sampling to learn
which memories are worth storing based on future utility.

Key features:
- Online learning (no batch retraining required)
- Thompson sampling for exploration
- Cold start with heuristic blending
- Integration with neuromodulator signals
- Low latency (<5ms per decision)

Usage:
    gate = LearnedMemoryGate(neuromod_orchestra=orchestra)

    # At encoding time
    decision = gate.predict(
        content_embedding=embedding,
        context=context,
        neuromod_state=neuromod_state
    )

    if decision.action == StorageDecision.STORE:
        # Store memory...
        gate.register_pending(memory_id, decision.features)

    # At outcome time
    gate.update(memory_id, utility=0.8)
"""

from __future__ import annotations

import logging
import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

from ww.core.memory_gate import GateContext, MemoryGate, StorageDecision

if TYPE_CHECKING:
    from ww.learning.neuromodulators import (
        NeuromodulatorOrchestra,
        NeuromodulatorState,
    )
    from ww.learning.three_factor import ThreeFactorLearningRule

logger = logging.getLogger(__name__)


@dataclass
class GateDecision:
    """Result of learned gate prediction."""

    action: StorageDecision
    probability: float  # P(useful | features)
    features: np.ndarray  # Feature vector for later training
    timestamp: datetime = field(default_factory=datetime.now)
    exploration_boost: float = 0.0  # Amount of exploration applied
    neuromod_state: NeuromodulatorState | None = None  # For three-factor learning


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


class LearnedMemoryGate:
    """
    Online learned memory gate using Bayesian logistic regression.

    Learns to predict P(memory will be useful) at encoding time,
    using delayed feedback from dopamine and serotonin systems.

    Algorithm: Thompson Sampling + Online Bayesian Logistic Regression
    - Model: p = σ(w^T φ + b)
    - Posterior: w ~ N(μ, Σ)
    - Update: Online gradient descent on log-loss

    Time complexity: O(d) per prediction/update (d = feature dimension)
    Space complexity: O(d²) for covariance (diagonal approx: O(d))
    """

    # Feature dimension breakdown
    # P0a: Learned content projection (1024 → 128) for task-specific features
    CONTENT_INPUT_DIM = 1024  # Raw BGE-M3 embedding dimension
    CONTENT_DIM = 128         # Projected content dimension (learned)
    CONTEXT_DIM = 64          # Project + task encoding
    NEUROMOD_DIM = 7          # DA, NE, 5-HT, ACh (one-hot), inhibition
    TEMPORAL_DIM = 16         # Time features
    INTERACTION_DIM = 32      # Cross products
    TOTAL_DIM = CONTENT_DIM + CONTEXT_DIM + NEUROMOD_DIM + TEMPORAL_DIM + INTERACTION_DIM  # 247

    def __init__(
        self,
        neuromod_orchestra: NeuromodulatorOrchestra | None = None,
        feature_dim: int = TOTAL_DIM,
        store_threshold: float = 0.6,
        buffer_threshold: float = 0.3,
        learning_rate_mean: float = 0.1,
        learning_rate_precision: float = 0.01,
        cold_start_threshold: int = 100,
        use_diagonal_covariance: bool = True,
        fallback_gate: MemoryGate | None = None,
        three_factor: ThreeFactorLearningRule | None = None,
        enable_three_factor: bool = False,
    ):
        """
        Initialize learned memory gate.

        Args:
            neuromod_orchestra: Neuromodulator system for signal integration
            feature_dim: Total feature dimension
            store_threshold: Probability threshold for immediate storage
            buffer_threshold: Probability threshold for buffering
            learning_rate_mean: Learning rate for weight mean (η)
            learning_rate_precision: Learning rate for weight precision (λ)
            cold_start_threshold: Number of observations before trusting model
            use_diagonal_covariance: Use diagonal approximation (faster, less memory)
            fallback_gate: Heuristic gate for cold start (created if None)
            three_factor: Optional ThreeFactorLearningRule for biologically-plausible learning
            enable_three_factor: If True, creates ThreeFactorLearningRule if not provided
        """
        self.neuromod = neuromod_orchestra

        # Three-factor learning integration
        self.three_factor = three_factor
        if enable_three_factor and self.three_factor is None:
            from ww.learning.three_factor import ThreeFactorLearningRule
            self.three_factor = ThreeFactorLearningRule()
        self.feature_dim = feature_dim
        self.θ_store = store_threshold
        self.θ_buffer = buffer_threshold
        self.η = learning_rate_mean
        self.λ = learning_rate_precision
        self.cold_start_threshold = cold_start_threshold
        self.use_diagonal = use_diagonal_covariance

        # Fallback heuristic gate
        self.fallback_gate = fallback_gate or MemoryGate()

        # Model parameters
        self.μ = self._init_weight_mean()
        self.Σ = self._init_weight_covariance()
        self.b = 0.0  # Bias (neutral prior)

        # Training state
        self.n_observations = 0
        # P0a: Now stores (features, raw_content_embedding, timestamp, neuromod_state) for projection learning
        # neuromod_state stored for three-factor learning modulation
        self.pending_labels: dict[UUID, tuple[np.ndarray, np.ndarray | None, datetime, NeuromodulatorState | None]] = {}

        # Statistics
        self.decisions = {"store": 0, "buffer": 0, "skip": 0}
        self.accuracy_buffer: deque[float] = deque(maxlen=1000)
        self.calibration_bins = np.linspace(0, 1, 11)  # For ECE computation
        self.calibration_counts = np.zeros(10)
        self.calibration_correct = np.zeros(10)

        # String embedding cache (for context encoding)
        # LEAK-001 FIX: Limit cache size to prevent memory exhaustion
        self._string_embed_cache: dict[str, np.ndarray] = {}
        self._string_embed_cache_max_size = 10000  # Max entries before eviction

        # P0a: Learned content projection (1024 → 128)
        # Xavier initialization for tanh activation
        self.W_content = np.random.randn(
            self.CONTENT_DIM, self.CONTENT_INPUT_DIM
        ).astype(np.float32) * np.sqrt(2.0 / (self.CONTENT_INPUT_DIM + self.CONTENT_DIM))
        self.b_content = np.zeros(self.CONTENT_DIM, dtype=np.float32)

        # Content projection learning rate (slower than main weights)
        self.η_content = learning_rate_mean * 0.1

        logger.info(
            f"LearnedMemoryGate initialized: dim={feature_dim}, "
            f"diagonal_cov={use_diagonal_covariance}, cold_start={cold_start_threshold}, "
            f"content_projection={self.CONTENT_INPUT_DIM}→{self.CONTENT_DIM}"
        )

    def _init_weight_mean(self) -> np.ndarray:
        """Initialize weight mean with informed priors."""
        μ = np.zeros(self.feature_dim)

        # Initialize content weights with random values for pattern discrimination
        # This allows immediate pattern discrimination (like DG sparse coding)
        # Scale gives content ~30% contribution to logit variance
        μ[:self.CONTENT_DIM] = np.random.randn(self.CONTENT_DIM) * 0.3

        # Indices for neuromodulator features
        neuromod_start = self.CONTENT_DIM + self.CONTEXT_DIM

        # Positive bias for:
        # - High dopamine RPE (surprising = valuable)
        # - High NE gain (novel = valuable)
        # - Encoding mode (should store)
        μ[neuromod_start + 0] = 0.5  # dopamine_rpe
        μ[neuromod_start + 1] = 0.3  # norepinephrine_gain
        μ[neuromod_start + 2] = 0.1  # serotonin_mood
        μ[neuromod_start + 3] = 0.4  # ach_encoding_mode (one-hot)
        μ[neuromod_start + 4] = 0.0  # ach_balanced_mode
        μ[neuromod_start + 5] = -0.2  # ach_retrieval_mode (negative: don't store)
        μ[neuromod_start + 6] = 0.0  # inhibition_sparsity

        return μ

    def _init_weight_covariance(self) -> np.ndarray:
        """Initialize weight covariance with uncertainty estimates."""
        if self.use_diagonal:
            # Diagonal covariance (store only variances as 1D array)
            Σ_diag = np.ones(self.feature_dim, dtype=np.float32) * 0.1

            # Higher uncertainty for content (need to learn)
            Σ_diag[:self.CONTENT_DIM] = 0.01

            # Medium uncertainty for context
            Σ_diag[self.CONTENT_DIM:self.CONTENT_DIM + self.CONTEXT_DIM] = 0.05

            # Higher uncertainty for neuromodulator interactions
            neuromod_start = self.CONTENT_DIM + self.CONTEXT_DIM
            Σ_diag[neuromod_start:neuromod_start + self.NEUROMOD_DIM] = 0.1

            # Return 1D array (diagonal elements only)
            return Σ_diag
        # Full covariance (identity scaled)
        return np.eye(self.feature_dim, dtype=np.float32) * 0.1

    def predict(
        self,
        content_embedding: np.ndarray,
        context: GateContext,
        neuromod_state: NeuromodulatorState,
        explore: bool = True
    ) -> GateDecision:
        """
        Predict storage decision with Thompson sampling exploration.

        Args:
            content_embedding: Content vector (1024-dim)
            context: Session context
            neuromod_state: Current neuromodulator state
            explore: If True, use Thompson sampling; else use mean

        Returns:
            GateDecision with action and metadata
        """
        # 1. Extract features
        φ = self._extract_features(content_embedding, context, neuromod_state)

        # 2. Sample weights (Thompson sampling for exploration)
        exploration_boost = 0.0
        if explore and self.n_observations > 0:
            # Boost exploration during learning phase
            if self.n_observations < 1000:
                exploration_factor = max(1.0, 3.0 - self.n_observations / 500)
                Σ_boosted = self.Σ * exploration_factor
                exploration_boost = exploration_factor - 1.0
            else:
                Σ_boosted = self.Σ

            # Additional boost from high NE (arousal-driven exploration)
            if neuromod_state.norepinephrine_gain > 1.5:
                Σ_boosted *= 1.5
                exploration_boost += 0.5

            # Sample from posterior
            if self.use_diagonal:
                # Diagonal: Σ is 1D array of variances
                std = np.sqrt(Σ_boosted)
                w = self.μ + np.random.randn(self.feature_dim) * std
            else:
                # Full: Σ is 2D covariance matrix
                w = np.random.multivariate_normal(self.μ, Σ_boosted)
        else:
            # Deterministic: use mean
            w = self.μ

        # 3. Compute probability
        logit = np.dot(w, φ) + self.b
        p_learned = float(sigmoid(logit))

        # 4. Cold start blending with heuristics
        if self.n_observations < self.cold_start_threshold:
            # Fallback to heuristic gate
            heuristic_result = self.fallback_gate.evaluate(
                content=str(content_embedding.mean()),  # Dummy content for scoring
                context=context
            )
            p_heuristic = heuristic_result.score

            # Blend: linear interpolation
            α = self.n_observations / self.cold_start_threshold
            p = (1 - α) * p_heuristic + α * p_learned

            logger.debug(
                f"Cold start blend: α={α:.2f}, "
                f"p_heur={p_heuristic:.3f}, p_learned={p_learned:.3f}, "
                f"p_final={p:.3f}"
            )
        else:
            p = p_learned

        # 5. ACh-modulated threshold adjustment
        θ_store_adj = self.θ_store
        if neuromod_state.acetylcholine_mode == "encoding":
            θ_store_adj *= 0.8  # Lower threshold (easier to store)
        elif neuromod_state.acetylcholine_mode == "retrieval":
            θ_store_adj *= 1.2  # Higher threshold (harder to store)

        # 6. Make decision
        if p >= θ_store_adj:
            action = StorageDecision.STORE
        elif p >= self.θ_buffer:
            action = StorageDecision.BUFFER
        else:
            action = StorageDecision.SKIP

        self.decisions[action.value] += 1

        return GateDecision(
            action=action,
            probability=p,
            features=φ,
            exploration_boost=exploration_boost,
            neuromod_state=neuromod_state
        )

    def _project_content(self, content_embedding: np.ndarray) -> np.ndarray:
        """
        Project raw content embedding to task-specific space.

        P0a: Learned projection 1024 → 128 with tanh activation.
        The projection learns to emphasize dimensions relevant for
        predicting memory utility.

        Args:
            content_embedding: Raw BGE-M3 embedding (1024-dim)

        Returns:
            Projected content features (128-dim)
        """
        # Ensure input is correct dimension
        if len(content_embedding) < self.CONTENT_INPUT_DIM:
            # Pad if needed
            content_embedding = np.pad(
                content_embedding,
                (0, self.CONTENT_INPUT_DIM - len(content_embedding))
            )
        elif len(content_embedding) > self.CONTENT_INPUT_DIM:
            # Truncate if needed
            content_embedding = content_embedding[:self.CONTENT_INPUT_DIM]

        # Linear projection with tanh activation
        projected = self.W_content @ content_embedding + self.b_content
        return np.tanh(projected)  # Bounded [-1, 1]

    def _update_content_projection(
        self,
        content_embedding: np.ndarray,
        error: float
    ) -> None:
        """
        Update content projection weights via backpropagation.

        Args:
            content_embedding: Raw content embedding used in forward pass
            error: Gradient signal from main loss (p - y)
        """
        # Get the projected value (before tanh)
        projected = self.W_content @ content_embedding + self.b_content
        tanh_out = np.tanh(projected)

        # tanh derivative: 1 - tanh^2
        tanh_grad = 1.0 - tanh_out ** 2

        # Gradient w.r.t. projection output (from main gate weights)
        # Simplified: use error magnitude scaled by content weight contribution
        content_grad = error * self.μ[:self.CONTENT_DIM] * tanh_grad

        # Update projection weights
        # grad_W = outer(content_grad, content_embedding)
        self.W_content -= self.η_content * np.outer(content_grad, content_embedding)
        self.b_content -= self.η_content * content_grad

    def _extract_features(
        self,
        content_embedding: np.ndarray,
        context: GateContext,
        neuromod_state: NeuromodulatorState
    ) -> np.ndarray:
        """
        Extract feature vector φ(x, c, n).

        Args:
            content_embedding: Content vector (1024-dim raw BGE-M3)
            context: Session context
            neuromod_state: Neuromodulator state

        Returns:
            Feature vector (feature_dim,) - now 247-dim with projection
        """
        # 1. Content features (128-dim via learned projection)
        content_features = self._project_content(content_embedding)

        # 2. Context features (64-dim)
        project_embed = self._embed_string(context.project or "default", dim=32)
        task_embed = self._embed_string(context.current_task or "general", dim=32)
        context_features = np.concatenate([project_embed, task_embed])

        # 3. Neuromodulator features (7-dim)
        neuromod_features = np.array([
            neuromod_state.dopamine_rpe,
            neuromod_state.norepinephrine_gain,
            neuromod_state.serotonin_mood,
            1.0 if neuromod_state.acetylcholine_mode == "encoding" else 0.0,
            1.0 if neuromod_state.acetylcholine_mode == "balanced" else 0.0,
            1.0 if neuromod_state.acetylcholine_mode == "retrieval" else 0.0,
            neuromod_state.inhibition_sparsity
        ], dtype=np.float32)

        # 4. Temporal features (16-dim)
        now = datetime.now()
        hour_sin = np.sin(2 * np.pi * now.hour / 24)
        hour_cos = np.cos(2 * np.pi * now.hour / 24)
        day_sin = np.sin(2 * np.pi * now.weekday() / 7)
        day_cos = np.cos(2 * np.pi * now.weekday() / 7)

        time_since_last = 0.0
        if context.last_store_time:
            elapsed = (now - context.last_store_time).total_seconds() / 3600
            time_since_last = float(np.clip(elapsed / 24, 0, 1))  # Normalize to [0, 1]

        msg_ratio = context.message_count_since_store / 20.0  # Normalize by max

        temporal_features = np.array([
            hour_sin, hour_cos, day_sin, day_cos,
            time_since_last, msg_ratio,
            *np.zeros(10)  # Reserved
        ], dtype=np.float32)

        # 5. Interaction features (32-dim) - nonlinear combinations
        content_mean = float(content_embedding.mean())
        content_std = float(content_embedding.std())

        interactions = np.array([
            content_mean * neuromod_state.dopamine_rpe,
            content_mean * neuromod_state.norepinephrine_gain,
            content_mean * neuromod_state.serotonin_mood,
            content_std * neuromod_state.dopamine_rpe,
            content_std * neuromod_state.norepinephrine_gain,
            *np.zeros(27)  # Reserved
        ], dtype=np.float32)

        # 6. Concatenate
        φ = np.concatenate([
            content_features,
            context_features,
            neuromod_features,
            temporal_features,
            interactions
        ])

        # Ensure correct dimension
        if len(φ) != self.feature_dim:
            logger.warning(
                f"Feature dimension mismatch: expected {self.feature_dim}, "
                f"got {len(φ)}, padding/truncating"
            )
            if len(φ) < self.feature_dim:
                φ = np.pad(φ, (0, self.feature_dim - len(φ)))
            else:
                φ = φ[:self.feature_dim]

        return φ

    def _embed_string(self, s: str, dim: int) -> np.ndarray:
        """
        Semantic string embedding using character n-gram hashing.

        P0b: Replaced pure hash-based random projection with n-gram based
        approach that captures semantic similarity - similar strings will
        have similar embeddings.

        Algorithm:
        1. Extract character 2-grams and 3-grams from normalized string
        2. Hash each n-gram to a position in the embedding
        3. Accumulate contributions (similar to feature hashing)
        4. Apply learned projection if available
        5. Normalize to unit sphere

        Args:
            s: String to embed
            dim: Embedding dimension

        Returns:
            Embedding vector with semantic properties
        """
        # Check cache
        cache_key = f"{s}:{dim}"
        if cache_key in self._string_embed_cache:
            return self._string_embed_cache[cache_key]

        # Normalize string
        s_norm = s.lower().strip()
        if not s_norm:
            s_norm = "empty"

        # Extract n-grams (2 and 3 character)
        ngrams = []

        # Character 2-grams
        for i in range(len(s_norm) - 1):
            ngrams.append(s_norm[i:i+2])

        # Character 3-grams
        for i in range(len(s_norm) - 2):
            ngrams.append(s_norm[i:i+3])

        # Word unigrams (for multi-word strings)
        words = s_norm.split()
        ngrams.extend(words)

        # Accumulate into embedding via feature hashing
        embedding = np.zeros(dim, dtype=np.float32)
        for ngram in ngrams:
            # Use hash to get position and sign
            h = hash(ngram)
            pos = abs(h) % dim
            sign = 1.0 if h >= 0 else -1.0
            embedding[pos] += sign * 1.0

        # Add base signal from full string hash (preserves exact match info)
        full_hash = hash(s_norm)
        base_pos = abs(full_hash) % dim
        embedding[base_pos] += 2.0  # Stronger signal for exact match

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding /= norm
        else:
            # Fallback for empty/degenerate case
            np.random.seed(hash(s) % (2**32))
            embedding = np.random.randn(dim).astype(np.float32)
            embedding /= np.linalg.norm(embedding)

        # Cache with LRU-style eviction (LEAK-001 FIX)
        if len(self._string_embed_cache) >= self._string_embed_cache_max_size:
            # Evict oldest entries (first 10% of cache)
            evict_count = self._string_embed_cache_max_size // 10
            for key in list(self._string_embed_cache.keys())[:evict_count]:
                del self._string_embed_cache[key]
        self._string_embed_cache[cache_key] = embedding

        return embedding

    def warm_string_cache_async(
        self,
        strings: list[str],
        dim: int,
        embedding_fn: Callable[..., Any]
    ) -> None:
        """
        Warm string embedding cache with BGE-M3 embeddings (optional).

        Call this asynchronously to pre-populate cache with high-quality
        semantic embeddings for known strings.

        Args:
            strings: List of strings to embed
            dim: Target dimension
            embedding_fn: Async function that returns full embeddings
        """
        # This is a hook for async warming - actual implementation would
        # call embedding_fn and project results to target dimension
        # For now, just pre-compute with n-gram method
        for s in strings:
            _ = self._embed_string(s, dim)

    def register_pending(
        self,
        memory_id: UUID,
        features: np.ndarray,
        raw_content_embedding: np.ndarray | None = None,
        neuromod_state: NeuromodulatorState | None = None
    ) -> None:
        """
        Register features for a memory awaiting feedback.

        Args:
            memory_id: ID of stored memory
            features: Feature vector used for prediction
            raw_content_embedding: Optional raw 1024-dim embedding for projection learning
            neuromod_state: Optional neuromodulator state for three-factor learning
        """
        # Clean up stale pending entries before registering new one
        self._cleanup_stale_pending()

        self.pending_labels[memory_id] = (features, raw_content_embedding, datetime.now(), neuromod_state)

        # Prune old pending labels (no feedback after 7 days = assume not useful)
        # RACE-001 FIX: Create copy of items to avoid "dict changed size during iteration"
        cutoff = datetime.now() - timedelta(days=7)
        expired = [
            mid for mid, (_, _, ts, _) in list(self.pending_labels.items())
            if ts < cutoff
        ]
        for mid in expired:
            features, raw_emb, ts, neuromod = self.pending_labels.pop(mid)
            # Treat as soft negative example (never retrieved within window)
            # Use 0.2 instead of 0.0 per Hinton: this conflates "genuinely not useful"
            # with "never queried for" - we should be less confident about negatives
            self._update_internal(features, utility=0.2, raw_content_embedding=raw_emb, neuromod_state=neuromod)
            logger.debug(f"Expired pending label for {mid}: treated as soft negative")

    def _cleanup_stale_pending(self) -> None:
        """
        Clean up stale pending entries older than 24h with neutral feedback.

        Any pending entry older than 24h (86400 seconds) gets neutral feedback (0.5)
        and is removed. This prevents unbounded growth of pending labels.
        """
        cutoff = datetime.now() - timedelta(hours=24)
        stale = [
            mid for mid, (_, _, ts, _) in list(self.pending_labels.items())
            if ts < cutoff
        ]
        for mid in stale:
            features, raw_emb, ts, neuromod = self.pending_labels.pop(mid)
            # Treat as neutral (0.5) - we don't know if it was useful or not
            self._update_internal(features, utility=0.5, raw_content_embedding=raw_emb, neuromod_state=neuromod)
            logger.debug(f"Stale pending label for {mid} (age: {(datetime.now() - ts).total_seconds() / 3600:.1f}h): treated as neutral")

    def update(self, memory_id: UUID, utility: float) -> None:
        """
        Update model after observing memory utility.

        Args:
            memory_id: Memory that was predicted for
            utility: Observed utility [0, 1]
        """
        # Retrieve stored features
        if memory_id not in self.pending_labels:
            logger.warning(f"No features found for memory {memory_id}")
            return

        φ, raw_content_emb, timestamp, neuromod_state = self.pending_labels.pop(memory_id)

        # Update model (including content projection if we have raw embedding)
        self._update_internal(φ, utility, raw_content_embedding=raw_content_emb, neuromod_state=neuromod_state)

        logger.debug(
            f"Updated gate for {memory_id}: utility={utility:.3f}, "
            f"n_obs={self.n_observations}"
        )

    def _update_internal(
        self,
        φ: np.ndarray,
        utility: float,
        raw_content_embedding: np.ndarray | None = None,
        neuromod_state: NeuromodulatorState | None = None
    ) -> None:
        """
        Internal update logic (online Bayesian logistic regression).

        Args:
            φ: Feature vector
            utility: Target utility [0, 1]
            raw_content_embedding: Optional raw embedding for projection update
            neuromod_state: Optional neuromodulator state for three-factor learning
        """
        # 1. Label smoothing to prevent overconfidence
        y = 0.1 + 0.8 * utility  # Maps [0, 1] → [0.1, 0.9]

        # 2. Current prediction
        logit = np.dot(self.μ, φ) + self.b
        p = float(sigmoid(logit))

        # 3. Compute gradient (binary cross-entropy)
        error = p - y
        grad_w = error * φ
        grad_b = error

        # 4. Compute effective learning rate (three-factor modulation if enabled)
        # LEARNING-HIGH-002 FIX: Use actual ThreeFactorLearningRule.compute() when available
        # Three-factor: effective_lr = base_lr * eligibility * neuromod_gate * dopamine_surprise
        effective_η = self.η
        if self.three_factor is not None:
            # Use proper three-factor learning computation
            from uuid import uuid4
            # Generate a temporary memory ID for this update (features don't have UUID)
            temp_id = uuid4()
            try:
                signal = self.three_factor.compute(
                    memory_id=temp_id,
                    base_lr=self.η,
                    outcome=utility,
                    neuromod_state=neuromod_state
                )
                effective_η = self.η * signal.effective_lr_multiplier
                logger.debug(
                    f"Three-factor LR: base={self.η:.4f}, "
                    f"eligibility={signal.eligibility:.2f}, "
                    f"neuromod_gate={signal.neuromod_gate:.2f}, "
                    f"dopamine={signal.dopamine_surprise:.2f}, "
                    f"effective={effective_η:.4f}"
                )
            except Exception as e:
                # Fallback to simple modulation if three-factor fails
                logger.warning(f"Three-factor compute failed: {e}, using fallback")
                if neuromod_state:
                    dopamine_factor = 0.5 + neuromod_state.dopamine_rpe
                    dopamine_factor = max(0.1, min(2.0, dopamine_factor))
                    ne_factor = 0.5 + 0.5 * neuromod_state.norepinephrine_gain
                    effective_η = self.η * dopamine_factor * ne_factor
        elif neuromod_state is not None:
            # Simplified modulation without ThreeFactorLearningRule
            # Use dopamine RPE as primary learning signal (prediction error)
            dopamine_factor = 0.5 + neuromod_state.dopamine_rpe  # Maps [-0.5, 0.5] → [0, 1]
            dopamine_factor = max(0.1, min(2.0, dopamine_factor))  # Bound [0.1, 2.0]

            # Norepinephrine modulates attention/salience
            ne_factor = 0.5 + 0.5 * neuromod_state.norepinephrine_gain

            # Combined modulation
            effective_η = self.η * dopamine_factor * ne_factor
            logger.debug(f"Simplified LR: base={self.η:.4f}, DA={dopamine_factor:.2f}, NE={ne_factor:.2f}, effective={effective_η:.4f}")

        # 5. Update precision (Sigma_inv <- Sigma_inv + lambda * phi * phi^T)
        if self.use_diagonal:
            # Diagonal approximation: Σ is 1D array
            sigma_inv_diag = 1.0 / self.Σ
            sigma_inv_diag += self.λ * (φ ** 2)
            self.Σ = 1.0 / sigma_inv_diag
        else:
            # Full update (expensive!)
            sigma_inv = np.linalg.inv(self.Σ)
            sigma_inv += self.λ * np.outer(φ, φ)
            self.Σ = np.linalg.inv(sigma_inv)

        # 6. Update mean (mu <- mu - effective_eta * Sigma * grad_w)
        if self.use_diagonal:
            # Σ is 1D array (diagonal elements)
            self.μ -= effective_η * self.Σ * grad_w
        else:
            # Σ is 2D matrix
            self.μ -= effective_η * (self.Σ @ grad_w)

        self.b -= effective_η * grad_b

        # 6. Update statistics
        self.n_observations += 1
        accuracy = 1.0 - abs(p - y)
        self.accuracy_buffer.append(accuracy)

        # Update calibration stats
        bin_idx = min(int(p * 10), 9)
        self.calibration_counts[bin_idx] += 1
        if abs(p - utility) < 0.2:  # Within 20% = "correct"
            self.calibration_correct[bin_idx] += 1

        # 7. P0a: Update content projection if raw embedding available
        if raw_content_embedding is not None:
            self._update_content_projection(raw_content_embedding, error)

    def batch_train(
        self,
        positive_samples: list[tuple[UUID, np.ndarray, float]],
        negative_samples: list[tuple[UUID, np.ndarray, float]],
        n_epochs: int = 1
    ) -> dict[str, float]:
        """
        Batch training on accumulated data (for periodic retraining).

        Args:
            positive_samples: List of (id, features, utility) for useful memories
            negative_samples: List of (id, features, utility) for not useful memories
            n_epochs: Number of passes over data

        Returns:
            Training statistics
        """
        all_samples = positive_samples + negative_samples
        random.shuffle(all_samples)

        losses = []
        for epoch in range(n_epochs):
            epoch_loss = 0.0

            for memory_id, φ, utility in all_samples:
                # Predict
                logit = np.dot(self.μ, φ) + self.b
                p = float(sigmoid(logit))

                # Target
                y = 0.1 + 0.8 * utility

                # Loss (BCE)
                loss = -(y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8))
                epoch_loss += loss

                # Update (same as online)
                error = p - y
                self.μ -= self.η * error * φ
                self.b -= self.η * error

            epoch_loss /= len(all_samples)
            losses.append(epoch_loss)

        self.n_observations += len(all_samples)

        return {
            "n_positives": len(positive_samples),
            "n_negatives": len(negative_samples),
            "final_loss": losses[-1] if losses else 0.0,
            "avg_loss": float(np.mean(losses)) if losses else 0.0
        }

    def get_stats(self) -> dict[str, float]:
        """Get gate statistics for monitoring."""
        total_decisions = sum(self.decisions.values())

        # Compute expected calibration error (ECE)
        ece = 0.0
        if self.calibration_counts.sum() > 0:
            for i in range(10):
                if self.calibration_counts[i] > 0:
                    bin_accuracy = self.calibration_correct[i] / self.calibration_counts[i]
                    bin_confidence = (i + 0.5) / 10  # Bin center
                    bin_weight = self.calibration_counts[i] / self.calibration_counts.sum()
                    ece += bin_weight * abs(bin_confidence - bin_accuracy)

        # Compute uncertainty trace (sum of variances)
        if self.use_diagonal:
            uncertainty_trace = float(np.sum(self.Σ))
        else:
            uncertainty_trace = float(np.trace(self.Σ))

        return {
            "n_observations": self.n_observations,
            "n_pending": len(self.pending_labels),
            "total_decisions": total_decisions,
            "store_rate": self.decisions["store"] / total_decisions if total_decisions > 0 else 0.0,
            "buffer_rate": self.decisions["buffer"] / total_decisions if total_decisions > 0 else 0.0,
            "skip_rate": self.decisions["skip"] / total_decisions if total_decisions > 0 else 0.0,
            "avg_accuracy": float(np.mean(self.accuracy_buffer)) if self.accuracy_buffer else 0.0,
            "expected_calibration_error": ece,
            "weight_norm": float(np.linalg.norm(self.μ)),
            "uncertainty_trace": uncertainty_trace,
            "cold_start_progress": min(1.0, self.n_observations / self.cold_start_threshold)
        }

    def save_state(self) -> dict:
        """
        Save gate state for persistence.

        Returns:
            Dictionary containing all learnable parameters and statistics.
        """
        return {
            # Content projection weights (P0a)
            "W_content": self.W_content.tolist(),
            "b_content": self.b_content.tolist(),
            "η_content": self.η_content,

            # Bayesian logistic regression parameters
            "μ": self.μ.tolist(),
            "Σ": self.Σ.tolist(),
            "b": self.b,
            "use_diagonal": self.use_diagonal,

            # Learning state
            "n_observations": self.n_observations,
            "decisions": self.decisions.copy(),

            # Calibration data
            "calibration_counts": self.calibration_counts.tolist(),
            "calibration_correct": self.calibration_correct.tolist(),

            # Config
            "cold_start_threshold": self.cold_start_threshold,
        }

    def load_state(self, state: dict) -> None:
        """
        Load gate state from saved dictionary.

        Args:
            state: Previously saved state dictionary.
        """
        # Content projection weights (P0a)
        if "W_content" in state:
            self.W_content = np.array(state["W_content"], dtype=np.float32)
        if "b_content" in state:
            self.b_content = np.array(state["b_content"], dtype=np.float32)
        if "η_content" in state:
            self.η_content = state["η_content"]

        # Bayesian logistic regression parameters
        if "μ" in state:
            self.μ = np.array(state["μ"], dtype=np.float32)
        if "Σ" in state:
            self.Σ = np.array(state["Σ"], dtype=np.float32)
        if "b" in state:
            self.b = state["b"]
        if "use_diagonal" in state:
            self.use_diagonal = state["use_diagonal"]

        # Learning state
        if "n_observations" in state:
            self.n_observations = state["n_observations"]
        if "decisions" in state:
            self.decisions = state["decisions"].copy()

        # Calibration data
        if "calibration_counts" in state:
            self.calibration_counts = np.array(state["calibration_counts"], dtype=np.int32)
        if "calibration_correct" in state:
            self.calibration_correct = np.array(state["calibration_correct"], dtype=np.int32)

        # Config
        if "cold_start_threshold" in state:
            self.cold_start_threshold = state["cold_start_threshold"]

    def reset(self) -> None:
        """Reset to initial state (for testing/debugging)."""
        self.μ = self._init_weight_mean()
        self.Σ = self._init_weight_covariance()
        self.b = 0.0
        self.n_observations = 0
        self.pending_labels.clear()
        self.decisions = {"store": 0, "buffer": 0, "skip": 0}
        self.accuracy_buffer.clear()
        self.calibration_counts.fill(0)
        self.calibration_correct.fill(0)


__all__ = [
    "GateDecision",
    "LearnedMemoryGate",
]
